import os
from app import app, db
from models import User, Student, Teacher, Schedule

def setup_system():
    print("🛑 Deleting Old Database...")
    
    # 1. Remove attendance.db if exists (Root folder)
    if os.path.exists('attendance.db'):
        os.remove('attendance.db')
        print("✅ Deleted 'attendance.db'")
        
    # 2. Remove instance/attendance.db if exists (Instance folder)
    if os.path.exists('instance/attendance.db'):
        os.remove('instance/attendance.db')
        print("✅ Deleted 'instance/attendance.db'")

    print("🛠️  Creating Fresh Database...")
    
    # 3. Create Fresh Tables
    with app.app_context():
        db.create_all()
        
        # --- 1. RESTORE SAHIL (Student) ---
        print("👤 Restoring Student: Sahil Rajesh Gaikawad...")
        sahil = Student(
            name="SAHIL RAJESH GAIKAWAD",
            roll_no="MCA-2025-011",
            email="sahilgaikawad03@gmail.com",
            course="MCA",
            phone="9876543210" 
        )
        sahil_user = User(
            name="SAHIL RAJESH GAIKAWAD",
            email="sahilgaikawad03@gmail.com",
            password="123", 
            role="student"
        )
        db.session.add(sahil)
        db.session.add(sahil_user)

        # --- 2. CREATE DUMMY TEACHER ---
        print("👩‍🏫 Creating Teacher: Prof. Anderson...")
        teacher = Teacher(
            name="Prof. Anderson",
            email="teacher@college.edu",
            phone="1234567890"
        )
        teacher_user = User(
            name="Prof. Anderson",
            email="teacher@college.edu",
            password="admin", 
            role="teacher"
        )
        db.session.add(teacher)
        db.session.add(teacher_user)

        # --- 3. CREATE DUMMY SCHEDULE ---
        print("📅 Scheduling Lecture for MCA...")
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        
        schedule = Schedule(
            teacher_email="teacher@college.edu",
            teacher_name="Prof. Anderson",
            subject="Data Structures (DSCC)",
            course="MCA",
            classroom="Lab-1",
            date=today,
            start_time="09:00",
            end_time="17:00"
        )
        db.session.add(schedule)

        db.session.commit()
        print("\n✅ SUCCESS! Database Fixed.")
        print("👉 Teacher Login: teacher@college.edu | Pass: admin")

if __name__ == "__main__":
    setup_system()