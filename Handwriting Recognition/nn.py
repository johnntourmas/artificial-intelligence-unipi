import tensorflow as tf  # θα μας βοηθήσει για να δημιουργήσουμε το neural network
import pandas as pd      # θα μας βοηθήσει για το διάβασμα δεδομένων και την επεξεργασία τους
from sklearn.model_selection import train_test_split  # θα μας βοηθήσει για να χωρίζουμε τα δεδομένα

# διαβάζουμε τα δεδομένα μας απο τα αρχείο
df = pd.read_csv('Handwritten Data.csv')
# ξέρουμε ότι η πρωτη στήλη είναι οι αριθμοί απο 0-25 που
# κωδικοποιούν το λατινικό αλφάβητο (δηλ. 0=Α, 1=Β κτλ)
# τα αποθηκεύουμε σε μία μεταβλητή x, και στοιχεία για κάθε γράμμα
# τα αποθηκεύουμε σε μια μεταβλητή y
x, y = df.drop(columns=['0']), df['0']

# χωρίζουμε τα δεδομένα σε train_letters και test_letters
# τα label περιέχουν αριθμούς απο 0-25, για καθε γράμμα της αλφαβήτου
# και τα train_letters/test_letters περιέχουν πληροφορίες για φωτογραφίες των 28*28 pixels
train_letters, test_letters, train_labels, test_labels = train_test_split(x, y)

# επεξεργασια δεδομενων
# για να προπονήσουμε ή να τεστάρουμε τα data, θα πρέπει τα δεδομένα μας
# να αναπαρηστάνονται σε νούμερα απο 0 έως 1. Κάθε pixel ξέρουμε ότι αναπαρηστάνεται από
# ένα νούμερο 0-255 οπότε κάθε νούμερο το διαιρούμε με 255 έτσι ώστε να πάρουμε ένα νούμερο από 0-1
train_letters = train_letters / 255.0
test_letters = test_letters / 255.0

# κατασκευή μοντέλου
# χρησιμοποιούμε ένα sequential μοντέλο διότι τα sequential μοντέλα
# είναι κατάλληλα όταν κάθε layer έχει μία είσοδο (στη συγκεκριμένη περίπτωση εικόνα)
# και μία έξοδο (στη συγκεκριμένη περίπτωση ένα γράμμα)
model = tf.keras.Sequential([
    # Για input layer χρησιμοποιούμε το flatten layer me input shape
    # τα pixel της εικόνας (28*28).
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    # το dense layer ή full connected layer είναι ενα κρυφό στρώμα που έχουμε
    # επιλέξει πως θα έχει 200 νευρώνες και για activation function να έχει τη relu
    tf.keras.layers.Dense(200, activation='relu'),  # hidden layer (2)
    # το output layer είναι και αυτό dense, έχει 26 νευρώνες (έναν για κάθε
    # γράμμα της αλφαβήτου) και έχουμε επιλέξει για activation function την softmax
    tf.keras.layers.Dense(26, activation='softmax')  # output layer (3)
])

# compile μοντέλου
# για optimizer χρησιμοποιύμε τον adam, και για υπολογισμό του loss του
# μοντέλου μας χρησιμοπούμε τη μέθοδο sparse_categorical_crossentropy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# εκπαιδεύοτμε το μοντελο μας πάνω στα δεδομένα μας,
# το εκπαιδεύουμε 2 φορες
model.fit(train_letters, train_labels, epochs=2)

# υπολογιίζουμε το loss και το accuracy πάνω στα test data που κρατήσαμε
loss, accuracy = model.evaluate(test_letters, test_labels)
print(f"loss: {loss}")
print(f"accuracy: {accuracy}")

# αποθηκέυουμε το μοντέλο για να το χρησιμοποιήσουμε μετά πάνω στα γράμματα
# model.save('recognize_letters.model')
