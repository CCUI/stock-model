from flask import Flask, render_template, request, jsonify
from datetime import datetime
from main import main
import threading
import queue
import logging

app = Flask(__name__)

# Configure Flask logger to only show warnings and errors
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

# Global queue for status updates
progress_queue = queue.Queue()

# Global variable to store the latest status
current_progress = {
    'step': '',
    'status': 'idle'
}

def run_analysis(market, include_news_sentiment, analysis_date):
    """Run the analysis in a separate thread and update status"""
    try:
        # Update status for initialization
        progress_queue.put({
            'step': 'Initializing analysis...',
            'status': 'running'
        })

        # Run the main analysis
        predictions, report = main(
            market=market,
            analysis_date=analysis_date,
            include_news_sentiment=include_news_sentiment
        )

        # Update status for completion
        progress_queue.put({
            'step': 'Analysis complete',
            'status': 'complete',
            'report': report
        })

    except Exception as e:
        # Handle errors
        progress_queue.put({
            'step': f'Error: {str(e)}',
            'status': 'error'
        })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get parameters from request
    market = request.form.get('market', 'UK')
    include_news = request.form.get('include_news') == 'true'
    date_str = request.form.get('date')

    # Parse date if provided
    analysis_date = None
    if date_str:
        try:
            analysis_date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format'}), 400

    # Reset status
    global current_progress
    current_progress = {
        'step': 'Starting analysis...',
        'status': 'running'
    }

    # Start analysis in a separate thread
    thread = threading.Thread(
        target=run_analysis,
        args=(market, include_news, analysis_date)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'started'})

@app.route('/progress')
def get_progress():
    """Get the current status of the analysis"""
    try:
        # Check for new status updates
        while not progress_queue.empty():
            global current_progress
            current_progress = progress_queue.get_nowait()

        return jsonify(current_progress)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)