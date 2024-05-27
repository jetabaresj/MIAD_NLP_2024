import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import neattext as nfx

MRL = joblib.load('./models/movie_genre_MRL.pkl')
tfidf = joblib.load('./models/Xfeat.pkl')

cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

def predict(event, context):
    try:

        body = json.loads(event['body'])
        plot = body['plot']

        plot_df = pd.DataFrame([plot], columns=['plot'])
        plot_df['plot'] = plot_df['plot'].apply(nfx.remove_stopwords)

        Xfeatures = tfidf.transform(plot_df['plot']).toarray()
        probabilities = MRL.predict_proba(Xfeatures)

        genre_df = pd.DataFrame(probabilities, columns=cols)
        genre_df_transposed = genre_df.transpose().rename(columns={0: 'Proba'})

        response = {'statusCode': 200, 'body': json.dumps(genre_df_transposed.to_dict())}
        return response
    
    except Exception as e:
        error_response = {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
        return error_response
