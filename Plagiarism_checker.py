import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get a list of text files in the current directory with the '.txt' extension
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]

# Read the content of each text file and store it in the student_notes list
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]

# Function to vectorize the text using TF-IDF (Term Frequency-Inverse Document Frequency)
def vectorize(Text):
    return TfidfVectorizer().fit_transform(Text).toarray()

# Function to calculate the cosine similarity between two documents
def similarity(doc1, doc2):
    return cosine_similarity([doc1, doc2])

# Vectorize the student notes using TF-IDF
vectors = vectorize(student_notes)

# Create a list of tuples where each tuple contains the filename and its corresponding vector
s_vectors = list(zip(student_files, vectors))

# Set to store plagiarism results
plagiarism_results = set()

# Function to check for plagiarism
def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            # Calculate the similarity score between student A and student B
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            #Turn into percentage..
            sim_percentage = sim_score * 100
            # Sort the student pairs to ensure consistent results
            student_pair = sorted((student_a, student_b))
            # Create a tuple with student names and their similarity score
            score = (student_pair[0], student_pair[1], sim_percentage)
            # Add the score to the plagiarism results set
            plagiarism_results.add(score)
    return plagiarism_results

# Call the check_plagiarism function and print the results
for data in check_plagiarism():
    print(data)
