from flask import Flask, render_template, request, jsonify, send_file
import requests
import urllib.parse
import re
import time
import pandas as pd
import io
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_single_smiles_to_iupac(smiles):
    """Convert single SMILES to IUPAC name"""
    try:
        smiles = smiles.strip()
        if not smiles:
            return {"input": smiles, "output": None, "error": "Empty SMILES string"}
        
        # Get CID from SMILES
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/cids/JSON"
        data = {"smiles": smiles}
        
        response = requests.post(url, data=data, timeout=20)
        if response.status_code != 200:
            return {"input": smiles, "output": None, "error": f"API error: {response.status_code}"}
        
        data = response.json()
        cids = data.get('IdentifierList', {}).get('CID', [])
        
        if not cids:
            return {"input": smiles, "output": None, "error": "No compound found"}
        
        cid = cids[0]
        
        # Get IUPAC name from CID
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/JSON"
        response = requests.get(url, timeout=20)
        
        if response.status_code != 200:
            return {"input": smiles, "output": None, "error": f"Failed to get IUPAC name: {response.status_code}"}
        
        data = response.json()
        properties = data.get('PropertyTable', {}).get('Properties', [])
        
        if not properties:
            return {"input": smiles, "output": None, "error": "No properties found"}
        
        iupac_name = properties[0].get('IUPACName')
        if not iupac_name:
            return {"input": smiles, "output": None, "error": "IUPAC name not available"}
        
        return {"input": smiles, "output": iupac_name, "error": None}
        
    except Exception as e:
        return {"input": smiles, "output": None, "error": f"Error: {str(e)}"}

def process_single_iupac_to_smiles(iupac_name):
    """Convert single IUPAC name to SMILES"""
    try:
        iupac_name = iupac_name.strip()
        if not iupac_name:
            return {"input": iupac_name, "output": None, "error": "Empty IUPAC name"}
        
        # URL encode the IUPAC name
        encoded_name = urllib.parse.quote(iupac_name)
        
        # Get CID from IUPAC name
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/cids/JSON"
        response = requests.get(url, timeout=20)
        
        if response.status_code != 200:
            return {"input": iupac_name, "output": None, "error": f"API error: {response.status_code}"}
        
        data = response.json()
        cids = data.get('IdentifierList', {}).get('CID', [])
        
        if not cids:
            return {"input": iupac_name, "output": None, "error": "No compound found"}
        
        cid = cids[0]
        
        # Get SMILES from CID
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
        response = requests.get(url, timeout=20)
        
        if response.status_code != 200:
            return {"input": iupac_name, "output": None, "error": f"Failed to get SMILES: {response.status_code}"}
        
        data = response.json()
        properties = data.get('PropertyTable', {}).get('Properties', [])
        
        if not properties:
            return {"input": iupac_name, "output": None, "error": "No properties found"}
        
        smiles = properties[0].get('CanonicalSMILES')
        if not smiles:
            return {"input": iupac_name, "output": None, "error": "SMILES not available"}
        
        return {"input": iupac_name, "output": smiles, "error": None}
        
    except Exception as e:
        return {"input": iupac_name, "output": None, "error": f"Error: {str(e)}"}

def batch_convert(input_list, conversion_func, max_workers=3, delay=0.5):
    """Batch convert list of inputs with rate limiting"""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_input = {
            executor.submit(conversion_func, item): item 
            for item in input_list
        }
        
        for future in as_completed(future_to_input):
            try:
                result = future.result()
                results.append(result)
                time.sleep(delay)  # Rate limiting to avoid overwhelming PubChem
            except Exception as e:
                input_item = future_to_input[future]
                results.append({
                    "input": input_item, 
                    "output": None, 
                    "error": f"Processing error: {str(e)}"
                })
    
    return results

def validate_input_list(input_text, input_type):
    """Validate and parse input list"""
    lines = [line.strip() for line in input_text.split('\n') if line.strip()]
    
    if not lines:
        return None, "No valid input provided"
    
    if len(lines) > 100:
        return None, "Maximum 100 compounds per batch allowed"
    
    # Basic validation
    valid_lines = []
    for line in lines:
        if input_type == 'smiles':
            if validate_smiles(line):
                valid_lines.append(line)
        else:  # iupac
            if validate_iupac(line):
                valid_lines.append(line)
    
    if not valid_lines:
        return None, "No valid inputs found"
    
    return valid_lines, None

def validate_smiles(smiles):
    """Basic SMILES validation"""
    if not smiles or len(smiles) < 2:
        return False
    
    # Basic pattern check
    pattern = r'^[A-Za-z0-9@+\-\[\]\(\)\\\/%=#\.]+$'
    return bool(re.match(pattern, smiles))

def validate_iupac(iupac_name):
    """Basic IUPAC validation"""
    if not iupac_name or len(iupac_name) < 3:
        return False
    
    # Should contain letters
    return any(c.isalpha() for c in iupac_name)

def parse_csv_file(file_path, input_type):
    """Parse CSV file and extract compounds"""
    try:
        df = pd.read_csv(file_path)
        compounds = []
        
        # Try to find the right column
        possible_columns = ['smiles', 'iupac', 'compound', 'name', 'input', 'molecule']
        
        for col in possible_columns:
            if col in df.columns:
                compounds = df[col].dropna().astype(str).tolist()
                break
        
        # If no specific column found, use first column
        if not compounds and len(df.columns) > 0:
            compounds = df.iloc[:, 0].dropna().astype(str).tolist()
        
        # Filter and validate compounds
        valid_compounds = []
        for compound in compounds:
            compound = str(compound).strip()
            if compound:
                if input_type == 'smiles' and validate_smiles(compound):
                    valid_compounds.append(compound)
                elif input_type == 'iupac' and validate_iupac(compound):
                    valid_compounds.append(compound)
        
        if not valid_compounds:
            return None, "No valid compounds found in CSV file"
        
        if len(valid_compounds) > 100:
            return None, "Maximum 100 compounds per batch allowed"
        
        return valid_compounds, None
        
    except Exception as e:
        return None, f"Error parsing CSV file: {str(e)}"

def create_csv_output(results, conversion_type):
    """Create CSV output from results"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    if conversion_type == 'smiles_to_iupac':
        writer.writerow(['SMILES', 'IUPAC_Name', 'Status', 'Error'])
        for result in results:
            writer.writerow([
                result['input'],
                result['output'] or '',
                'Success' if result['output'] else 'Failed',
                result['error'] or ''
            ])
    else:
        writer.writerow(['IUPAC_Name', 'SMILES', 'Status', 'Error'])
        for result in results:
            writer.writerow([
                result['input'],
                result['output'] or '',
                'Success' if result['output'] else 'Failed',
                result['error'] or ''
            ])
    
    return output.getvalue()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/batch_convert', methods=['POST'])
def batch_convert_route():
    conversion_type = request.form.get('conversion_type')
    input_text = request.form.get('input_text', '').strip()
    input_type = request.form.get('input_type', 'smiles')
    
    if not input_text:
        return jsonify({
            'success': False,
            'error': 'Please enter input text'
        })
    
    # Parse and validate input list
    input_list, error = validate_input_list(input_text, input_type)
    if error:
        return jsonify({'success': False, 'error': error})
    
    try:
        if conversion_type == 'smiles_to_iupac':
            results = batch_convert(input_list, process_single_smiles_to_iupac)
        else:  # iupac_to_smiles
            results = batch_convert(input_list, process_single_iupac_to_smiles)
        
        # Create CSV output
        csv_output = create_csv_output(results, conversion_type)
        
        # Count statistics
        successful = sum(1 for r in results if r['output'] is not None)
        failed = len(results) - successful
        
        return jsonify({
            'success': True,
            'results': results,
            'csv_data': csv_output,
            'statistics': {
                'total': len(results),
                'successful': successful,
                'failed': failed
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Batch processing error: {str(e)}'
        })

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'csv_file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['csv_file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        conversion_type = request.form.get('conversion_type')
        input_type = 'smiles' if conversion_type == 'smiles_to_iupac' else 'iupac'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Parse CSV file
        input_list, error = parse_csv_file(file_path, input_type)
        if error:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'success': False, 'error': error})
        
        try:
            if conversion_type == 'smiles_to_iupac':
                results = batch_convert(input_list, process_single_smiles_to_iupac)
            else:  # iupac_to_smiles
                results = batch_convert(input_list, process_single_iupac_to_smiles)
            
            # Create CSV output
            csv_output = create_csv_output(results, conversion_type)
            
            # Count statistics
            successful = sum(1 for r in results if r['output'] is not None)
            failed = len(results) - successful
            
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({
                'success': True,
                'results': results,
                'csv_data': csv_output,
                'statistics': {
                    'total': len(results),
                    'successful': successful,
                    'failed': failed
                }
            })
            
        except Exception as e:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({
                'success': False,
                'error': f'Batch processing error: {str(e)}'
            })
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/download_csv', methods=['POST'])
def download_csv():
    csv_data = request.form.get('csv_data')
    conversion_type = request.form.get('conversion_type')
    
    if not csv_data:
        return jsonify({'success': False, 'error': 'No data to download'})
    
    # Create in-memory file
    mem = io.BytesIO()
    mem.write(csv_data.encode('utf-8'))
    mem.seek(0)
    
    filename = f"conversion_results_{conversion_type}.csv"
    
    return send_file(
        mem,
        as_attachment=True,
        download_name=filename,
        mimetype='text/csv'
    )

@app.route('/get_examples', methods=['GET'])
def get_examples():
    examples = {
        'smiles': [
            'CCO',
            'CC(=O)O',
            'C1=CC=CC=C1',
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
        ],
        'iupac': [
            'ethanol',
            'acetic acid',
            'benzene',
            '1,3,7-trimethylpurine-2,6-dione',
            'ibuprofen'
        ]
    }
    return jsonify(examples)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)