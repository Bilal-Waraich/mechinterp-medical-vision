import os
import random
import pandas as pd
from PIL import Image, ImageDraw

def generate_shape_dataset(output_dir='shapes_data', num_samples=100, img_size=(64, 64)):
    os.makedirs(output_dir, exist_ok=True)
    data = []
    shapes = ['circle', 'square', 'triangle']
    
    for i in range(num_samples):
        # Create a black background
        img = Image.new('RGB', img_size, color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Randomize shape, color, and coordinates
        shape_type = random.choice(shapes)
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        
        # Define random bounding box for the shape
        m = 10 # margin
        x1, y1 = random.randint(m, 20), random.randint(m, 20)
        x2, y2 = random.randint(40, img_size[0]-m), random.randint(40, img_size[1]-m)
        
        if shape_type == 'circle':
            draw.ellipse([x1, y1, x2, y2], fill=color)
        elif shape_type == 'square':
            draw.rectangle([x1, y1, x2, y2], fill=color)
        elif shape_type == 'triangle':
            points = [(random.randint(x1, x2), y1), (x1, y2), (x2, y2)]
            draw.polygon(points, fill=color)
            
        # Save image and record label
        file_path = f"{output_dir}/shape_{i}.png"
        img.save(file_path)
        data.append({'file_path': file_path, 'label': shape_type})
        
    # Save labels to CSV
    pd.DataFrame(data).to_csv('labels.csv', index=False)
    print(f"Generated {num_samples} images in '{output_dir}/' and saved labels to 'labels.csv'.")

generate_shape_dataset()