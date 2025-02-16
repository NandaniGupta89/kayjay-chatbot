# kayjay-chatbot

## Setup Instructions

### 1. Create and Activate a Virtual Environment

Open a terminal and navigate to the project directory. Then, run the following commands:

```sh
python -m venv venv
```

#### Windows:
```sh
venv\Scripts\activate
```

### 2. Install Dependencies

Run the following command to install the required dependencies:

```sh
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root directory and add your Google Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

### 4. Run the Application

Use Streamlit to start the chatbot application:

```sh
streamlit run app.py
```


