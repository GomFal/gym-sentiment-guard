from transformers import pipeline

classifier = pipeline(
    task='sentiment-analysis', model='lxyuan/distilbert-base-multilingual-cased-sentiments-student'
)

texts = ['Me encanta este gimnasio', 'El servicio es terrible', 'Es un sitio normal, nada especial']

results = classifier(texts)

for t, r in zip(texts, results):
    print(t, '->', r)
