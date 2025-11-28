import pickle
import os
from app import app
from models import db, Student, User

# File Path for Face Data
PICKLE_DB_PATH = 'embeddings/insight_db.pkl'

def delete_student_data(target_roll_no):
    print(f"\n🗑️  Attempting to delete: {target_roll_no}")
    print("="*50)

    # -------------------------------------------
    # 1. DELETE FROM PICKLE FILE (Face Data)
    # -------------------------------------------
    if os.path.exists(PICKLE_DB_PATH):
        try:
            # Load Data
            with open(PICKLE_DB_PATH, 'rb') as f:
                faces_data = pickle.load(f)
            
            # Check if exists and Delete
            if target_roll_no in faces_data:
                del faces_data[target_roll_no]
                
                # Save Back
                with open(PICKLE_DB_PATH, 'wb') as f:
                    pickle.dump(faces_data, f)
                print(f"✅ Deleted Face Data from Pickle for: {target_roll_no}")
            else:
                print(f"⚠️  Roll No '{target_roll_no}' not found in Pickle file.")
        except Exception as e:
            print(f"❌ Error reading pickle file: {e}")
    else:
        print("❌ Pickle file not found.")

    # -------------------------------------------
    # 2. DELETE FROM DATABASE (Text Data)
    # -------------------------------------------
    with app.app_context():
        try:
            # Find Student by Roll No
            student = Student.query.filter_by(roll_no=target_roll_no).first()
            
            if student:
                # User login bhi delete karna padega (agar email same hai)
                user = User.query.filter_by(email=student.email).first()
                if user:
                    db.session.delete(user)
                    print(f"✅ Deleted Login Account for: {student.email}")

                # Delete Student Record
                db.session.delete(student)
                db.session.commit()
                print(f"✅ Deleted Database Record for: {target_roll_no}")
            else:
                print(f"⚠️  Student '{target_roll_no}' not found in Database.")
                
        except Exception as e:
            print(f"❌ Database Error: {e}")

    print("="*50 + "\n")

if __name__ == "__main__":
    # Yahan user se input lenge
    roll_to_delete = input("Enter Roll Number/Name to delete: ").strip()
    
    if roll_to_delete:
        delete_student_data(roll_to_delete)
        print("Done.")
    else:
        print("Operation Cancelled.")