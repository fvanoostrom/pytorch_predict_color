import os
import cv2
import numpy as np

class ColorRenderer:
    def __init__(self, color_table):
        self.color_table = color_table

    def generate_color_image(self):
        num_colors = len(self.color_table)
        num_columns = int(np.ceil(np.sqrt(num_colors)))
        num_rows = int(np.ceil(num_colors / num_columns))
        box_size = 60
        image_width = num_columns * box_size
        image_height = num_rows * box_size

        # Create a blank white image
        image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

        for i, color in enumerate(self.color_table):
            bgr_value = tuple(reversed(color["rgb_value"]))
            closest_bgr_value = tuple(reversed(color["closest_rgb_value"]))
            closest_color_name = color["closest_color_name"]
            predicted_bgr_value = tuple(reversed(color["predicted_rgb_value"]))
            predicted_color_name = color["predicted_color_name"]
            row, col = divmod(i, num_columns)
            y_start, y_end = row * box_size, (row + 1) * box_size
            x_start, x_end = col * box_size, (col + 1) * box_size
            text_color = (0, 0, 0) if predicted_color_name == closest_color_name else (0, 0, 255)

            cv2.rectangle(image, (x_start, y_start), ((x_start + x_end) // 2, (y_start + y_end) // 2), bgr_value, -1)
            cv2.rectangle(image, ((x_start + x_end) // 2, y_start), (x_end, (y_start + y_end) // 2), closest_bgr_value, -1)
            cv2.putText(image, closest_color_name, (x_start + 5, (y_start + y_end) // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)

            cv2.rectangle(image, (x_start, (y_start + y_end) // 2), (x_end, y_end), predicted_bgr_value, -1)
            cv2.putText(image, predicted_color_name, (x_start + 5, y_end - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)

        return image

    def save_image(self, image, save_path="output/color_table_output.png"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image)
        print(f"Color table image saved at: {save_path}")

    def display_image(self, image):
        cv2.imshow('Color Table', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def render_color_table(self, save_path=None):
        color_image = self.generate_color_image()
        if save_path:
            self.save_image(color_image, save_path)
        self.display_image(color_image)
