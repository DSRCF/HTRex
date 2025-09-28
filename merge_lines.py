import numpy as np

def merge_text_lines(results, x_tolerance=30, y_tolerance_scale=0.5):
    """
    Merges bounding boxes that are on the same horizontal line with improved logic.
    
    Args:
        results: The raw output from PaddleOCR.
        x_tolerance: Max horizontal distance between boxes to be considered for merging.
        y_tolerance_scale: Scale factor for vertical tolerance based on box height.
        
    Returns:
        A list of merged bounding boxes.
    """
    if not results or not results[0]:
        return []

    # Extract boxes and sort them primarily by top-y, secondarily by left-x
    boxes = sorted([line[0] for line in results[0]], key=lambda x: (x[0][1], x[0][0]))

    merged_lines = []
    if not boxes:
        return merged_lines

    current_line = [boxes[0]]
    for box in boxes[1:]:
        last_box_in_line = current_line[-1]
        
        # Calculate vertical centers
        last_box_y_center = (last_box_in_line[0][1] + last_box_in_line[2][1]) / 2
        current_box_y_center = (box[0][1] + box[2][1]) / 2
        
        # Calculate vertical tolerance based on the height of the taller box
        box_height = max(
            abs(box[0][1] - box[2][1]),
            abs(last_box_in_line[0][1] - last_box_in_line[2][1])
        )
        y_tolerance = box_height * y_tolerance_scale

        # Calculate horizontal distance (gap between boxes)
        last_box_x_max = max(p[0] for p in last_box_in_line)
        current_box_x_min = min(p[0] for p in box)
        horizontal_gap = current_box_x_min - last_box_x_max

        # --- Improved Merging Condition ---
        # Merge if vertically aligned AND horizontally close.
        if abs(current_box_y_center - last_box_y_center) < y_tolerance and horizontal_gap < x_tolerance:
            current_line.append(box)
        else:
            # New line detected, finalize the previous one
            all_points = np.concatenate(current_line, axis=0)
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)
            merged_lines.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
            
            # Start a new line
            current_line = [box]

    # Merge the last line
    if current_line:
        all_points = np.concatenate(current_line, axis=0)
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        merged_lines.append([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

    return merged_lines