"""
Clean up old test data from vector database
"""

import shutil
from pathlib import Path
import sys

def check_running_processes():
    """Check for processes that might be using the database."""
    try:
        import psutil
        
        print("🔍 Checking for running processes...")
        python_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    if proc.info['cmdline']:
                        cmdline_str = ' '.join(proc.info['cmdline']).lower()
                        if any(keyword in cmdline_str for keyword in ['enhanced_app', 'smartrag', 'fastapi']):
                            python_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        if python_processes:
            print(f"⚠️  Found {len(python_processes)} Python processes that might be using the database:")
            for proc in python_processes:
                try:
                    cmdline = ' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else 'Unknown'
                    print(f"   PID: {proc.pid} - {cmdline}")
                except:
                    print(f"   PID: {proc.pid} - Process info unavailable")
            
            print("\n🛑 Please stop the web app first, then run this script again.")
            print("   You can stop it by pressing Ctrl+C in the terminal running the enhanced_app.py")
            return False
        
        print("✅ No conflicting processes found")
        return True
        
    except ImportError:
        print("⚠️  psutil not available - skipping process check")
        print("   Make sure to stop any running web apps manually")
        return True

def clean_vector_database():
    """Remove the entire vector database to start fresh."""
    print("🧹 Cleaning Vector Database")
    print("=" * 40)
    
    # Check for running processes first
    if not check_running_processes():
        return False
    
    vector_db_path = Path("./vector_db")
    
    if vector_db_path.exists():
        try:
            # Remove entire vector database directory
            shutil.rmtree(vector_db_path)
            print("✅ Successfully removed vector database directory")
            print("📁 Path removed:", vector_db_path.absolute())
            
            # Verify it's gone
            if not vector_db_path.exists():
                print("✅ Cleanup confirmed - vector database is now empty")
            else:
                print("❌ Cleanup failed - directory still exists")
                return False
                
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            print("\n🔧 Manual cleanup steps:")
            print("1. Stop all Python processes")
            print("2. Delete the 'vector_db' folder manually")
            print("3. Restart the Streamlit app")
            return False
    else:
        print("ℹ️  Vector database directory doesn't exist - already clean")
    
    # Also offer to clean user data
    user_data_path = Path("user_data")
    if user_data_path.exists():
        try:
            response = input("\n🤔 Do you also want to clear user data (uploaded files history and query history)? (y/N): ")
            if response.lower() in ['y', 'yes']:
                shutil.rmtree(user_data_path)
                print("✅ User data cleared!")
            else:
                print("ℹ️  User data preserved")
        except:
            print("ℹ️  User data preserved")
    
    print("\n🎯 Next steps:")
    print("1. Start your Streamlit app: python run_streamlit.py")
    print("2. Upload your files")
    print("3. The system will create a fresh, clean vector database")
    return True

if __name__ == "__main__":
    clean_vector_database()
