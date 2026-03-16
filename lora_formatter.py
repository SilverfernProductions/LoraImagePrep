import os
import threading
import requests
import torch
import FreeSimpleGUI as sg
from bs4 import BeautifulSoup
from PIL import Image
from urllib.parse import urljoin
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Global AI Variables ---
processor = None
model = None

def load_ai_model(window, values):
    """Loads the BLIP model in a background thread."""
    global processor, model
    try:
        window["-LOG-"].print("Initializing AI Model... (First run downloads ~900MB)")
        window["-PROG-"].update(20, 100)
        
        model_id = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(model_id)
        
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
        
        window["-PROG-"].update(100, 100)
        window["-LOG-"].print(f"AI Model Loaded successfully on {device.upper()}.")
        window.write_event_value("-MODEL-LOADED-", True)
    except Exception as e:
        window["-LOG-"].print(f"Model Load Error: {e}")
        window.write_event_value("-MODEL-LOADED-", False)

def generate_caption(image_path):
    """Generates AI description for an image."""
    try:
        raw_image = Image.open(image_path).convert('RGB')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = processor(raw_image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return ""

def process_logic(values, window):
    """The core logic for downloading and processing images."""
    folder = values["-FOLDER-"]
    url = values["-URL-"]
    out_dir = values["-OUT_FOLDER-"] # New dynamic output folder
    trigger = values["-TRIGGER-"]
    do_cap = values["-DOCAPTION-"]
    res = int(values["-RES-"])
    max_img = int(values["-MAX_IMG-"]) if values["-MAX_IMG-"].isdigit() else 999
    
    if not out_dir:
        out_dir = "lora_training_data"
        
    os.makedirs(out_dir, exist_ok=True)
    
    # --- 1. Web Scraper ---
    if url:
        window["-LOG-"].print(f"Scraping URL: {url}...")
        temp_dir = "temp_downloads"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            found_imgs = [urljoin(url, i.get("src")) for i in soup.find_all("img") if i.get("src")]
            
            valid_imgs = [u for u in found_imgs if any(u.lower().split('?')[0].endswith(e) for e in ['.jpg', '.jpeg', '.png', '.webp'])]
            valid_imgs = valid_imgs[:max_img]
            
            if not valid_imgs:
                window["-LOG-"].print("No valid images found at URL.")
            else:
                window["-LOG-"].print(f"Downloading {len(valid_imgs)} images...")
                for i, u in enumerate(valid_imgs):
                    window["-PROG-"].update(i + 1, len(valid_imgs))
                    img_data = requests.get(u, timeout=10).content
                    with open(os.path.join(temp_dir, f"web_{i}.jpg"), "wb") as f:
                        f.write(img_data)
                folder = temp_dir
        except Exception as e:
            window["-LOG-"].print(f"URL Download Error: {e}")

    # --- 2. Image Formatting & AI Captioning ---
    if folder and os.path.isdir(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        total = len(files)
        
        if total == 0:
            window["-LOG-"].print("No images found to process.")
            return

        window["-LOG-"].print(f"Processing {total} images...")
        
        for i, filename in enumerate(files):
            window["-PROG-"].update(i + 1, total)
            window["-LOG-"].print(f"[{i+1}/{total}] Formatting: {filename}")
            
            input_path = os.path.join(folder, filename)
            try:
                with Image.open(input_path) as img:
                    img = img.convert("RGB")
                    img.thumbnail((res, res), Image.Resampling.LANCZOS)
                    new_img = Image.new("RGB", (res, res), (255, 255, 255))
                    offset = ((res - img.size[0]) // 2, (res - img.size[1]) // 2)
                    new_img.paste(img, offset)
                    
                    base_name = os.path.splitext(filename)[0]
                    save_path = os.path.join(out_dir, f"{base_name}.jpg")
                    new_img.save(save_path, "JPEG", quality=95)
                    
                    # Generate Caption
                    final_caption = trigger
                    if do_cap and model:
                        description = generate_caption(input_path)
                        if description:
                            final_caption = f"{trigger}, {description}"
                    
                    with open(os.path.join(out_dir, f"{base_name}.txt"), "w", encoding="utf-8") as f:
                        f.write(final_caption)
            except Exception as e:
                window["-LOG-"].print(f"Error processing {filename}: {e}")

        window["-LOG-"].print(f"COMPLETE! Files saved in: {os.path.abspath(out_dir)}")
        sg.popup("Task Finished", f"Processed {total} images successfully.\nLocation: {out_dir}")
    else:
        window["-LOG-"].print("Error: No valid source folder found.")

# --- GUI Layout ---
layout = [
    [sg.Text("Lora Image Formatter Enhanced (LIFE)", font=("Helvetica", 20), text_color="lightblue")],
    [sg.Text("Source Folder:"), sg.Input(key="-FOLDER-"), sg.FolderBrowse()],
    [sg.Text("Web URL:      "), sg.Input(key="-URL-"), sg.Text("Limit:"), sg.Input("20", key="-MAX_IMG-", size=(5,1))],
    
    [sg.Text("Output Folder:"), sg.Input("lora_training_data", key="-OUT_FOLDER-"), sg.FolderBrowse()],
    
    [sg.HSeparator(pad=(0, 15))],
    [sg.Text("Trigger Word: "), sg.Input("ohwx", key="-TRIGGER-", size=(15,1)), 
     sg.Checkbox("Use AI Auto-Captioning", default=True, key="-DOCAPTION-")],
    [sg.Text("Resolution:   "), sg.Combo([256,512, 768, 1024], default_value=1024, key="-RES-")],
    
    [sg.Button("Start Processing", size=(20, 1), button_color=("white", "green")), sg.Button("Exit")],
    [sg.Text("Progress:")],
    [sg.ProgressBar(100, orientation='h', size=(40, 20), key="-PROG-")],
    [sg.Multiline(size=(80, 12), key="-LOG-", autoscroll=True, disabled=True, font=("Consolas", 10))]
]

window = sg.Window("LIFE", layout)
model_loading_started = False

while True:
    event, values = window.read(timeout=100)
    
    if event in (sg.WIN_CLOSED, "Exit"):
        break
    
    if event == "Start Processing":
        if values["-DOCAPTION-"] and model is None:
            if not model_loading_started:
                model_loading_started = True
                threading.Thread(target=load_ai_model, args=(window, values), daemon=True).start()
            else:
                window["-LOG-"].print("AI Model is still loading...")
        else:
            threading.Thread(target=process_logic, args=(values, window), daemon=True).start()

    if event == "-MODEL-LOADED-":
        model_loading_started = False
        if values["-MODEL-LOADED-"]:
            threading.Thread(target=process_logic, args=(values, window), daemon=True).start()
        else:
            sg.popup_error("Failed to load AI model.")

window.close()