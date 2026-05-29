import gymnasium as gym
import torch
import numpy as np
import os
from datetime import datetime


# טעינת המודל מתוך קובץ הסטודנטים
from d3qn_lunar_baseline import DuelingQNetwork 

def evaluate_lunar_lander(model_path, video_folder="./lunar_videos"):
    """
    Loads the trained LunarLander model and records a full video of the flight.
    """
    # 1. יצירת סביבת ה-Lander במצב הקלטת וידאו
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    # 2. עטיפת הסביבה במקליט וידאו לפרק הראשון (אינדקס 0)
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder=video_folder, 
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix = f"flight_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    
    # 3. אתחול וטעינת נתונים
    state, info = env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 4. טעינת הרשת והמשקולות
    model = DuelingQNetwork(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() 
    
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    
    print(f"Starting Lunar Lander flight evaluation...")
    print(f"Recording flight video into '{video_folder}'...")
    
    # 5. לולאת הסימולציה
    while not (done or truncated):
        # בחירת פעולה חמדתית (Greedy)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()
        
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        total_reward += reward
        steps += 1
        
    env.close()
    
    # 6. סיכום התוצאות למסך
    print("\n" + "="*45)
    print("             LUNAR LANDER SUMMARY            ")
    print("="*45)
    
    # בסביבה זו, ציון מעל 200 פירושו נחיתה מושלמת ומצליחה
    if total_reward >= 200:
        print("🚀 Status: SUCCESSFUL LANDING! 🎉 (Score >= 200)")
    elif done and total_reward < 0:
        print("💥 Status: CRASHED! (Hard Impact)")
    else:
        print("🛸 Status: Finished flight (Incomplete or unstable landing)")
        
    print(f"📊 Total Flight Reward: {total_reward:.2f}")
    print(f"⏱️ Total Steps in Air:  {steps}")
    print(f"📁 Video File Saved To:  {os.path.abspath(video_folder)}")
    print("="*45)

if __name__ == "__main__":
    weights_path = "d3qn_lunar_model.pth"
    
    if os.path.exists(weights_path):
        evaluate_lunar_lander(model_path=weights_path)
    else:
        print(f"Error: Could not find '{weights_path}'. Please run your training script first.")