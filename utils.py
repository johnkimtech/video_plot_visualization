import cv2
import os

def create_video_from_images(input_folder, output_video_path, frame_rate=1.0):
    # Get all JPEG files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    # Sort the files based on frame index (assuming the format img_000x.jpg)
    image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Create VideoWriter object
    frame = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Iterate through images, add text, and write to video
    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)

        # Extract frame index from the filename
        frame_index = int(image_file.split('_')[-1].split('.')[0])

        # Add text to the image
        text = f'Frame: {frame_index}'
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the image to the video
        out.write(img)

    # Release VideoWriter
    out.release()

if __name__ == "__main__":
    # Example usage:
    input_folder = '/media/hoang/423D-B2A0/draw_sbj/0022128380/0022128380_1'
    output_video_path = 'output_video.avi'
    create_video_from_images(input_folder, output_video_path, frame_rate=30.0)
