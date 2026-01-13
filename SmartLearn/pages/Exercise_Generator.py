import streamlit as st
import requests
import os

st.set_page_config(page_title="Exercise Generator", page_icon="📝")

# Check Access
# Check Access - Removed
if 'user' not in st.session_state:
    st.session_state.user = "Guest"


st.title("📝 Exercise Generator")
st.markdown("Generate practice exercises tailored to your needs.")

# API Configuration
EXERCISE_API_URL = os.getenv("EXERCISE_API_URL", "http://localhost:5001")

# Fetch Subjects
@st.cache_data
def get_subjects():
    try:
        response = requests.get(f"{EXERCISE_API_URL}/subjects", timeout=5)
        if response.status_code == 200:
            return response.json().get("subjects", [])
    except:
        pass
    return ["Algebra", "Calculus", "Trigonometry", "Geometry", "Statistics"]

subjects_list = get_subjects()

with st.sidebar:
    st.header("Configuration")
    subject = st.selectbox("Subject", subjects_list)
    difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"])
    num_questions = st.number_input("Number of Questions", min_value=1, max_value=10, value=3)
    
    generate_btn = st.button("Generate Exercises", type="primary")

if generate_btn:
    with st.spinner("Generating exercises via AI..."):
        try:
            payload = {
                "subject": subject,
                "difficulty": difficulty.lower(),
                "count": num_questions
            }
            response = requests.post(f"{EXERCISE_API_URL}/generate", json=payload, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.exercises = data.get("exercises", [])
                st.success(f"Generated {len(st.session_state.exercises)} exercises!")
            else:
                st.error(f"Failed to generate exercises: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to Exercise Generator service. Is it running?")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if 'exercises' in st.session_state and st.session_state.exercises:
    st.subheader("Practice Time")
    
    for exercise in st.session_state.exercises:
        ex_id = exercise.get('id')
        st.divider()
        st.markdown(f"**Q{ex_id}:** {exercise.get('question', '')}")
        
        # Hint button
        hint_key = f"hint_{ex_id}"
        if st.button(f"Show Hint for Q{ex_id}", key=f"hint_btn_{ex_id}"):
            st.session_state[hint_key] = True
        if st.session_state.get(hint_key, False):
            st.info(f"💡 **Hint:** {exercise.get('hint', 'No hint available.')}")
        
        # Solution button
        solution_key = f"solution_{ex_id}"
        if st.button(f"Show Solution for Q{ex_id}", key=f"solution_btn_{ex_id}"):
            st.session_state[solution_key] = True
        if st.session_state.get(solution_key, False):
            with st.expander(f"**Detailed Solution for Q{ex_id}**"):
                st.write(exercise.get('solution', 'No solution provided.'))
                st.markdown(f"**Correct Answer:** `{exercise.get('answer', '')}`")
        
        # Answer input and submit
        answer_key = f"answer_{ex_id}"
        submit_key = f"submit_{ex_id}"
        result_key = f"result_{ex_id}"
        
        user_answer = st.text_input(f"Your Answer for Q{ex_id}", key=answer_key)
        
        if st.button(f"Submit Answer for Q{ex_id}", key=f"submit_btn_{ex_id}"):
            if not user_answer.strip():
                st.warning(f"Please enter an answer for Q{ex_id}.")
            else:
                st.session_state[submit_key] = True
                # Check answer
                try:
                    check_payload = {
                        "expected": exercise.get('answer', ''),
                        "student": user_answer.strip()
                    }
                    check_resp = requests.post(f"{EXERCISE_API_URL}/check-answer", json=check_payload, timeout=10)
                    
                    if check_resp.status_code == 200:
                        is_correct = check_resp.json().get("correct", False)
                    else:
                        is_correct = user_answer.strip().lower() == exercise.get('answer', '').lower()
                        
                except:
                    is_correct = user_answer.strip().lower() == exercise.get('answer', '').lower()
                
                st.session_state[result_key] = {
                    "correct": is_correct,
                    "user_answer": user_answer.strip(),
                    "correct_answer": exercise.get('answer', ''),
                    "solution": exercise.get('solution', 'No solution provided.')
                }
        
        if st.session_state.get(submit_key, False):
            result = st.session_state[result_key]
            if result["correct"]:
                st.success(f"Q{ex_id}: Correct! ✅")
            else:
                st.error(f"Q{ex_id}: Incorrect. ❌")
                st.info(f"Your answer: `{result['user_answer']}`")
                st.info(f"Correct answer: `{result['correct_answer']}`")
            with st.expander(f"**Detailed Solution for Q{ex_id}**"):
                st.write(result["solution"])
