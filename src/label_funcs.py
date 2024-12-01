# Label extraction functions

def binary_gender_label(filename):
    # Infer gender class from filename
    # e.g.  'F_30_1234.jpg' -> 0
    #       'M_30_1234.jpg' -> 1
    label = filename.split('_')[0].upper()
    label = (0 if label == 'F'
            else 1 if label == 'M'
            else None)
    return label

def age_label_male(filename):
    if binary_gender_label(filename) != 1: return None
    label = int(filename.split('_')[1])
    return label

def age_label_female(filename):
    if binary_gender_label(filename) != 0: return None
    label = int(filename.split('_')[1])
    return label

def age_label_all(filename):
    label = int(filename.split('_')[1])
    return label

def class_age_label(class_bounds):
    # this works but not tested properly due to time constraints
    def f(filename):
        for c, class_min in enumerate(class_bounds):
            if age_label_all(filename) < class_min:
                return c-1
    return f