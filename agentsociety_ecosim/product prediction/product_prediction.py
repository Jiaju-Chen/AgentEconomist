import pandas as pd
import networkx as nx
from tqdm import tqdm
from openai import OpenAI

import os 
api_key = os.environ.get('OPENAI_API_KEY')

class ProductTimeline():
    def __init__(self, company_name):
        self.company_name = company_name
        # file of company product timeline
        self.company_file = company_name + '.csv'
        self.timeline = pd.read_csv(self.company_file, engine='python', encoding='utf-8')
        # use native Llama 70B
        self.client = OpenAI(base_url="http://35.220.164.252:3888/v1/", api_key=api_key)

    def get_response(self, prompt):
        response = self.client.chat.completions.create(
            model="USD-guiji/deepseek-v3",
            messages=[
                {"role": "system", "content": "你是一个金融领域的AI助手。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10240,
            temperature=0.4,
            top_p=0.97
        )
        return response.choices[0].message.content

    def predict_type(self):
        # predict the number of new product for each product type
        prompt = '''I'll give you some data about new product development in Intel Corp. each year from 2014 to 2020.
                The input of each year is year numbers, following several lines describing the number of new product developed in each type.
                The format is $product type path$: $new product number$
                You are going to predict the number of new product in each type from 2021 to 2023.
                Please make sure the output follows the same format as input, and the product type path to be predicted should only come from the path that appeared in the input.
                now the input are as follows:
                '''
        look_back = ''
        predict = ''
        for y, t_year in self.timeline.groupby('Year'):
            type_count = t_year.groupby('Path').size()
            if int(y) <= 2020:
                look_back += 'Year {}\n'.format(y)
                for t, c in type_count.items():
                    look_back += '{}: {}\n'.format(t, c)
            else:
                predict += 'Year {}\n'.format(y)
                for t, c in type_count.items():
                    predict += '{}: {}\n'.format(t, c)

        prompt += look_back
        prompt += '''\nPlease ONLY output the predictions, no other descriptions.'''
        return self.get_response(prompt)

    def predict_description(self):
        # predict the description of each new product
        ds_prompt = '''I'll give you some data about new product development in Intel, Corp. each year from 2014 to 2020.
        The input of each year is year numbers, following several lines describing the information of each new product.
        The format of each product information is 3 sentences, containing product name, product description and product classification path.
        You are going to predict the description of each new product from 2021 to 2023.
        now the input are as follows:
        '''
        ds_look_back = ''
        ds_predict = ''
        for y, t_year in self.timeline.groupby('Year'):
            if int(y) <= 2020:
                ds_look_back += 'Year {}\n'.format(y)
                for _, row in t_year.iterrows():
                    ds_look_back += 'The company launched a new product {}. The description is {}. We think the classification is {}\n'.format(
                        row['Product Name'], row['Product Description'], row['Path'])
            else:
                ds_predict += 'Year {}\n'.format(y)
                for t, d_year in t_year.groupby('Path'):
                    ds_predict += '***{}: {}\n'.format(t, len(d_year))
                    for _, row in d_year.iterrows():
                        ds_predict += '{}\n'.format(row['Product Description'])
        ds_prompt += ds_look_back
        number_predict = self.predict_type()
        ds_prompt += 'You have predicted the number of product for each category as follows:\n {}'.format(
            number_predict)
        ds_prompt += '''
        now you need to predcit the product description for each category.
        The output should be:
        Year 2021
        ***$category1$: $number of product$
        $product1 description$
        $product2 description$
        ***$category2$: $number of product$
        $product1 description$
        $product2 description$
        ...

        Year 2022
        ***$category1$: $number of product$
        $product1 descritions$
        ...
        ...
        Please make sure the output follows the format described above, no other words. And do not output product name in each product description, only the attribute.
        '''
        return self.get_response(ds_prompt)
    

if __name__ == "__main__":
    company_name = 'Intel Corp.v2'
    pt = ProductTimeline(company_name)
    print('Product Type Prediction:')
    print(pt.predict_type())
    print('\nProduct Description Prediction:')
    print(pt.predict_description())