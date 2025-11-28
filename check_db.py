import os
import pickle
from app import app
# Note: Teacher bhi import kiya
from models import db, Student, User, Teacher

def inspect_data():
    print("\n" + "="*60)
    print("📂 STORAGE LOCATIONS (Yahan aapka data save hai)")
    print("="*60)
    
    # 1. Check SQLite DB Path
    db_file = "attendance.db"
    instance_db = os.path.join("instance", "attendance.db")
    
    if os.path.exists(db_file):
        print(f"✅ Text Database Found at: {os.path.abspath(db_file)}")
    elif os.path.exists(instance_db):
        print(f"✅ Text Database Found at: {os.path.abspath(instance_db)}")
    else:
        print("❌ Database file NOT found! (Run app.py first)")

    # 2. Check Pickle File Path
    pkl_path = os.path.join("embeddings", "insight_db.pkl")
    if os.path.exists(pkl_path):
        print(f"✅ Face Data Found at:   {os.path.abspath(pkl_path)}")
    else:
        print("❌ Face Data Pickle file NOT found!")

    with app.app_context():
        
        # --- SECTION 1: STUDENTS ---
        print("\n" + "="*60)
        print("🎓 STUDENT TABLE CONTENTS")
        print("="*60)
        students = Student.query.all()
        if not students:
            print("❌ No Students Registered yet.")
        else:
            print(f"{'Roll No':<15} {'Name':<20} {'Email':<25} {'Course'}")
            print("-" * 75)
            for s in students:
                print(f"{s.roll_no:<15} {s.name:<20} {s.email:<25} {s.course}")

        # --- SECTION 2: TEACHERS ---
        print("\n" + "="*60)
        print("👩‍🏫 TEACHER TABLE CONTENTS")
        print("="*60)
        teachers = Teacher.query.all()
        if not teachers:
            print("❌ No Teachers Registered yet.")
        else:
            print(f"{'ID':<5} {'Name':<20} {'Email':<25} {'Phone'}")
            print("-" * 65)
            for t in teachers:
                print(f"{t.id:<5} {t.name:<20} {t.email:<25} {t.phone if t.phone else 'N/A'}")

        # --- SECTION 3: USERS (LOGIN CREDENTIALS) ---
        print("\n" + "="*60)
        print("🔑 USER TABLE (Login Passwords)")
        print("="*60)
        users = User.query.all()
        if not users:
            print("❌ No Users Registered yet.")
        else:
            print(f"{'Role':<10} {'Email':<25} {'Password'}")
            print("-" * 50)
            for u in users:
                print(f"{u.role:<10} {u.email:<25} {u.password}")

    # --- SECTION 4: FACE DATA ---
    print("\n" + "="*60)
    print("👤 FACE BIOMETRICS DATA (Pickle File)")
    print("="*60)

    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            if not data:
                print("❌ Pickle file is empty.")
            else:
                print(f"Total Registered Faces: {len(data)}")
                print("-" * 30)
                for roll, embedding in data.items():
                    print(f"🆔 Roll No: {roll} -> ✅ Embedding Size: {len(embedding)}")
        except Exception as e:
            print(f"Error reading pickle: {e}")

    print("\n" + "="*60)

if __name__ == "__main__":
    inspect_data()