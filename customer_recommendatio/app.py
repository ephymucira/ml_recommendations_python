import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the pre-trained model
with open('recommendation_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

df = model_data['df']
cosine_sim = model_data['cosine_sim']
product_user_matrix = model_data['product_user_matrix']
hybrid_recommendation = model_data['hybrid_recommendation']


@app.route('/recommend', methods=['POST'])
def recommend_products():
    try:
        data = request.get_json()
        product_id = data['product_id']

        recommendations = hybrid_recommendation(product_id, cosine_sim, product_user_matrix, df)
        # Convert the recommendations DataFrame to a list of dictionaries
        recommendations_list = recommendations.to_dict(orient='records')
        return jsonify(recommendations_list)

    except KeyError:
        return jsonify({'error': 'product_id is required'}), 400  # Bad Request
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Internal Server Error


if __name__ == '__main__':
    app.run(debug=True)