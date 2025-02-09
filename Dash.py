import os
import subprocess
from flask import Flask, render_template, jsonify, send_from_directory
import pandas as pd

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# EEG Analysis Script Execution
def run_eeg_analysis():
    """Runs EEG analysis script if needed and returns success or failure."""
    try:
        print("Running EEG analysis script...")
        result = subprocess.run(['python', 'Analyse.py'], capture_output=True, text=True, check=True)
        print("Analyse.py Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("Error running Analyse.py:", e.stderr)
        return False

# Load and preprocess EEG data
def get_eeg_status(student_id):
    """Fetches EEG status and recommendation for a student."""
    try:
        # Run EEG analysis if needed
        if not run_eeg_analysis():
            return {"error": "EEG analysis failed."}

        # Read recommendations
        recommendations_file = 'eeg_recommendations.csv'
        if os.path.exists(recommendations_file):
            recommendations = pd.read_csv(recommendations_file)

            # Filter student-specific data
            student_recommendation = recommendations[recommendations['Student_ID'] == student_id]
            if not student_recommendation.empty:
                latest_recommendation = student_recommendation.iloc[-1]  # Latest entry for student
                return {
                    "student_id": student_id,
                    "time": latest_recommendation['Time'],
                    "recommendation": latest_recommendation['Recommendation']
                }
            else:
                return {"error": f"No data found for Student ID {student_id}"}
        else:
            return {"error": "Recommendation file not found"}
    
    except Exception as e:
        return {"error": str(e)}

# Home route - Serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Time Table route - Ensure the EEG analysis image is visible
@app.route('/TimeTable')
def timetable():
    # Image path in the static directory
    eeg_image_path = os.path.join(app.static_folder, 'images', 'eeg_analysis.png')
    
    # Check if the image exists
    if os.path.exists(eeg_image_path):
        eeg_image_url = '/static/images/eeg_analysis.png'  # Correct image path for the static folder
    else:
        eeg_image_url = None  # No image available
    
    return render_template('TimeTable.html', eeg_image_url=eeg_image_url)

# API route to get EEG analysis data (Image URL and Conclusion)
@app.route('/api/eeg_analysis')
def get_eeg_analysis():
    eeg_image_url = '/static/images/eeg_analysis.png'
    eeg_conclusion = "This is a sample EEG analysis conclusion."

    # Check if EEG image exists
    if not os.path.exists(os.path.join(app.static_folder, 'images', 'eeg_analysis.png')):
        return jsonify({"error": "EEG image not found"})

    return jsonify({
        "image_url": eeg_image_url,
        "conclusion": eeg_conclusion
    })

# Examination route
@app.route('/exam')
def exam():
    return render_template('exam.html')

# Change Password route
@app.route('/password')
def password():
    return render_template('password.html')

# API route to get EEG status for a student
@app.route('/api/student/<student_id>')
def get_student_eeg(student_id):
    eeg_status = get_eeg_status(student_id)
    return jsonify(eeg_status)

if __name__ == '__main__':
    app.run(debug=True)
