from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='templates')

# --- 1. TRAINING DATA ---
TRAINING_DATA = {
    'MEDICAL': 'medical medicine mbbs doctor surgery biology health nursing pharmacy hospital clinic dentist healthcare physician cardiology',
    'AI_TECH': 'artificial intelligence ai machine learning ml data science python automation robotics future technology neural networks tech',
    'DESIGN': 'graphic design fashion design creative arts textile clothes photoshop illustrator sketching branding ui ux designer logos artist',
    'LAW': 'law lawyer justice court legal advocate judge criminal law corporate law constitution politics administration litigation solicitor',
    'ENGINEERING': 'mechanical electrical civil engineering machines construction circuits power engines infrastructure robots manufacturing structural',
    'BUSINESS_FINANCE': 'business finance money investment stocks mba marketing entrepreneurship startup management profit accounting banking commerce'
}

# --- 2. CAREER DATABASE ---
CAREER_DETAILS = {
    'MEDICAL': {
        'Title': 'Medicine & Health Sciences',
        'Overview': 'A life-saving field focused on diagnosing and treating illnesses through clinical practice and research.',
        'Roadmap': 'High School -> MDCAT -> 5-Year MBBS -> 1-Year House Job -> Specialization.',
        'Salary_Scope': 'High stability and prestigious income potential.'
    },
    'AI_TECH': {
        'Title': 'Artificial Intelligence & Data Science',
        'Overview': 'The frontier of technology, creating systems that simulate human intelligence and solve complex data problems.',
        'Roadmap': 'High School -> BS CS/AI -> Mastering Python/Math -> Portfolio Building -> Industry Internship.',
        'Salary_Scope': 'Top-tier global salaries with massive remote work opportunities.'
    },
    'DESIGN': {
        'Title': 'Professional Design & Creative Arts',
        'Overview': 'A field combining aesthetics and functionality to create visual identities, fashion, and digital experiences.',
        'Roadmap': 'Intermediate -> Bachelor in Design (B.Des) -> Tool Mastery (Adobe/CAD) -> Portfolio Development.',
        'Salary_Scope': 'High freelance potential and roles in global fashion/tech brands.'
    },
    'LAW': {
        'Title': 'Law & Legal Studies',
        'Overview': 'The study of legal systems, justice, and corporate regulations to defend rights and manage disputes.',
        'Roadmap': 'Intermediate -> LAT Test -> 5-Year LL.B -> Bar Exam -> Legal License.',
        'Salary_Scope': 'Influential roles in judiciary, corporate firms, and private practice.'
    },
    'ENGINEERING': {
        'Title': 'Engineering (Mechanical/Electrical/Civil)',
        'Overview': 'The application of physics and math to build the physical and electronic infrastructure of the world.',
        'Roadmap': 'High School -> ECAT -> 4-Year B.E/BS Degree -> PEC Certification -> Industrial Role.',
        'Salary_Scope': 'Essential roles in manufacturing, energy, and construction sectors.'
    },
    'BUSINESS_FINANCE': {
        'Title': 'Business, Finance & Management',
        'Overview': 'The management of capital, resources, and organizations to drive economic growth and corporate success.',
        'Roadmap': 'High School -> BBA/BS Finance -> Professional Certs (ACCA/CFA/CA) or MBA -> Corporate Leadership.',
        'Salary_Scope': 'Lucrative careers in banking, audit firms, and global startups.'
    }
}

# --- 3. AI CLASSIFICATION ENGINE ---
labels = list(TRAINING_DATA.keys())
texts = list(TRAINING_DATA.values())

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X_train = vectorizer.fit_transform(texts)
clf = LinearSVC(dual=False)
clf.fit(X_train, labels)

# --- 4. LOGIC HANDLERS ---

def get_all_fields_summary():
    """Generates a brief summary for all categories."""
    summary = "### 🌍 **Global Career Overview**\n"
    summary += "Here is a brief look at all the fields I can help you with:\n\n"
    
    for key, data in CAREER_DETAILS.items():
        summary += f"🔹 **{data['Title']}**\n"
        summary += f"_{data['Overview']}_\n\n"
    
    summary += "--- \n*Type the name of any specific field above to see its full **Step-by-Step Roadmap**!*"
    return summary

def get_detailed_response(user_text):
    user_text = user_text.lower().strip() 
    
    # Check for "All Fields" intent
    if any(word in user_text for word in ['all fields', 'list', 'everything', 'summary']):
        return get_all_fields_summary()

    # Regular AI Prediction
    vec = vectorizer.transform([user_text])
    similarity = cosine_similarity(vec, X_train)
    
    if np.max(similarity) < 0.05:
        return "I couldn't identify a specific match. Please try keywords like **Medical, AI, Law, Design, Engineering, or Business**."

    prediction = clf.predict(vec)[0]
    data = CAREER_DETAILS.get(prediction)
    
    if data:
        resp = f"### 📌 **{data['Title']}**\n\n"
        resp += f"**Overview:** {data['Overview']}\n\n"
        resp += f"**Roadmap:** {data['Roadmap']}\n\n"
        resp += f"**Career Potential:** {data['Salary_Scope']}\n\n"
        return resp + "*Would you like a deeper breakdown of this path?*"
    
    return "Data for this field is currently being updated."

# --- 5. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    
    if any(greet in user_message.lower() for greet in ['hi', 'hello', 'hey', 'start']):
        return jsonify({'response': "Hello! I am your **AI Career Counselor**. I can summarize **All Fields** or give a detailed roadmap for a specific one. What is on your mind?"})

    response_text = get_detailed_response(user_message)
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)