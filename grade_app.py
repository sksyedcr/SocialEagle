
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Student Grade Manager",
    page_icon="📚",
    layout="centered",
)

# Keep student records in session_state so they survive Streamlit reruns.
if "students" not in st.session_state:
    st.session_state.students = []


def get_grade(mark):
    """Return the grade for a mark using the assignment grading scale."""
    if mark >= 90:
        return "A"
    elif mark >= 80:
        return "B"
    elif mark >= 70:
        return "C"
    elif mark >= 60:
        return "D"
    return "E"


st.title("Student Grade Manager")
st.write("Add student marks and view the class results.")

# Form: Streamlit submits the name and mark together.
with st.form("add_student_form", clear_on_submit=True):
    name = st.text_input("Name", placeholder="Enter student name")
    mark = st.number_input(
        "Mark",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
    )
    submitted = st.form_submit_button("Add student")

    if submitted:
        cleaned_name = name.strip()

        if not cleaned_name:
            st.error("Please enter a student name.")
        else:
            st.session_state.students.append(
                {
                    "Name": cleaned_name,
                    "Mark": int(mark),
                    "Grade": get_grade(int(mark)),
                }
            )
            st.success(f"{cleaned_name} added successfully.")

st.divider()
st.subheader("Class Results")

if st.session_state.students:
    # Display all students in a table.
    st.table(st.session_state.students)

    marks = [student["Mark"] for student in st.session_state.students]
    average = sum(marks) / len(marks)
    highest = max(marks)
    lowest = min(marks)

    # Display class statistics as Streamlit metrics.
    col1, col2, col3 = st.columns(3)
    col1.metric("Average", f"{average:.1f}")
    col2.metric("Highest", highest)
    col3.metric("Lowest", lowest)

    # Optional button to clear all records for a fresh run.
    if st.button("Clear all students"):
        st.session_state.students = []
        st.rerun()
else:
    st.info("No students added yet. Use the form above to add the first student.")
