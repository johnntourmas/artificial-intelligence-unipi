import numpy as np
import tensorflow as tf
import cv2

# δημιουργούμε ένα λεξικό με τα γράμματα των επιθέτων μας
surname_letters = {1: 'B',
                   13: 'N',
                   }

# κάνουμε load το μοντέλο που δημιουργλησαμε
model = tf.keras.models.load_model('recognize_letters.model')

# στο input βάζουμε το όνομα της εικόνας 
input_img = 'B1.png'
# φορτώνουμε την εικόνα σε κατάλληλη μορφή (gray scale)
img = cv2.imread(input_img)[:, :, 0]

# μετατρέπουμε την εικόνα σε array πίνακα για να μπορούμε
# να την εισάγουμε στο μοντέλο μας
img2 = np.invert(np.array([img]))

# κανουμε την πρόβλεψη και την εκτυπώνουμε 
prediction = model.predict(img2)

# αν ο αριθμός υπάρχει στο λεξικό εκτυπώνουμε την αντίστοιχή τιμή του λεξικού
# διαφορετικά εκτυπώνουμε το κωδικοποιημένο γράμμα
if int(np.argmax(prediction)) in surname_letters.keys():
    print(f"Input: {input_img}\nPrediction: {surname_letters[int(np.argmax(prediction))]}")
else:
    print(f"Input: {input_img}\nPrediction: {np.argmax(prediction)}")
