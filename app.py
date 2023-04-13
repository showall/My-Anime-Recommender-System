# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, MATCH, Patch
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc
from dash import dcc
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sub
import dash
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
from recommender import Recommender
import pickle
import random
import textwrap
# Incorporate data
df_anime = pd.read_csv('data/anime2.csv')
#print(len(df_anime))
output = []
for i in range (4):
    output.append(dbc.Col( [html.Img(src= 'https://raw.githubusercontent.com/michaelbabyn/plot_data/master/naphthalene.png' , style={'height':'80%',"width": "70%",
                                                                                                                                     "text-align":"center",
                                                                                                                                     "opacity":"0",
                                                                                                                                     })
    , html.P('anime name', style ={"text-align":"center"} )], style={'height':'10%',  "float": "left", "width": "25%","padding": "5px"}))
        
    




# Initialize the app - incorporate a Dash Mantine theme
external_stylesheets = [dmc.theme.DEFAULT_COLORS]
app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
server = app.server
app.layout = html.Div(style={
"background-image": 'linear-gradient(rgba(255,255,255,0.8), rgba(255,255,255,0.8)), url("assets/background.jpg")',} ,
    children= [dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Which Anime Should I Watch Next?", style={"text-align":"center", "color":"blue","margin-top":"3%"}),
                        html.H6("A Mini Recommendation Engine by Gerard Sho", style={"text-align":"left", "color":"black", "font-size":"10"}),
                    ],
                    width=True,
                ),
                # dbc.Col([
                #     html.Img(src="assets/MIT-logo-red-gray-72x38.svg", alt="MIT Logo", height="30px"),
                # ], width=1)
            ],
            align="end",
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H5("Key Parameters"),
                                dmc.RadioGroup(
                                        [dmc.Radio(i, value=i) for i in  ['New_Viewer', 'Regular_Viewer', 'Premium_Viewer']],
                                        id='user_type',
                                        value='New_Viewer', style={"margin": "5px", "font-size":"25px","padding-left":"14%"}
                                    ),


                                # html.P("Number of booms:"),
                                # dcc.Slider(
                                #     id="n_booms",
                                #     min=1,
                                #     max=3,
                                #     step=1,
                                #     value=3,
                                #     marks={1: "1", 2: "2", 3: "3",},
                                # ),
                                # html.P("Wing Span [m]:"),
                                # dcc.Input(id="wing_span", value=43, type="number"),
                                # html.P("Angle of Attack [deg]:"),
                                # dcc.Input(id="alpha", value=7.0, type="number"),


                            ], style={"background-color": "#eeeee4", "color": "black"},
                        ),
                        # html.Div(
                        #     [
                        #         html.H5("Commands"),

                        #         dbc.Button(
                        #             "LL Analysis (3s)",
                        #             id="run_ll_analysis",
                        #             color="secondary",
                        #             style={"margin": "5px"},
                        #             n_clicks_timestamp="0",
                        #         ),
                        #         dbc.Button(
                        #             "VLM Analysis (15s)",
                        #             id="run_vlm_analysis",
                        #             color="secondary",
                        #             style={"margin": "5px"},
                        #             n_clicks_timestamp="0",
                        #         ),
                        #     ]
                        # ),
                        html.Hr(),
                    ], style={"background-color": "#eeeee4", "color": "black"},
                    width=3,
                ),
                dbc.Col(
                    [
                        html.Div( [dcc.Dropdown(
                                 df_anime.name, clearable=True, placeholder = "Select Your Watched Anime",
                                multi=True, id="choose", value=None,  style={"height":"100", }),
                                    dbc.Button(
                                    "Find Recommendation",
                                    id="dynamic-add-filter-btn1", n_clicks=0,
                                 #   n_clicks_timestamp="0",
                                    color="primary",
                                    style={"margin": "5px","position": "relative", "left": "35%", "top": "35%"},
                                )],
                                 style={"height":"200"})
                                ,
                        dmc.Title('Recommended Just For You', color="black", size="h2"),
                        html.Div( id ="dynamic_image_1" , children=[] ),
                        dmc.Title('', color="green", size="h3"),                                

                        html.Div(children=[], id ="dynamic_image_0"), 
                #        html.Div(children=output), 
                        html.Hr(),
                        html.Br(),

                        html.Div([  
                        html.Hr(),
                        html.Br(),
                        dmc.Title('@gerardsho', color="blue", size="h6",style ={"position":"relative","margin-top":"200px", "padding-top":"20%", "text-align":"right"} ) ,
                        html.Div(children=[], id ="dynamic_image_2") ], style={"height":"300"}) ,
                        html.Hr(),      
                        html.Div([                            
                     #   dmc.Title('Welcome, New User', color="orange", size="h3"),
                        html.Div(children=[], id ="dynamic_image_3")    
                        ]),
                 
                    ],
                    width=True,
                ),
            ]
        ),
        html.Hr(),
        # html.P(
        #     [
        #         html.A(
        #             "Source code",
        #             href="https://github.com/peterdsharpe/AeroSandbox-Interactive-Demo",
        #         ),
        #         ". Aircraft design tools powered by ",
        #         html.A(
        #             "AeroSandbox", href="https://peterdsharpe.github.com/AeroSandbox"
        #         ),
        #         ". Build beautiful UIs for your scientific computing apps with ",
        #         html.A("Plot.ly ", href="https://plotly.com/"),
        #         "and ",
        #         html.A("Dash", href="https://plotly.com/dash/"),
        #         "!",
        #     ]
        # ),
    #     dmc.Title('My First App with Data, Graph, and Controls', color="blue", size="h3"),
    # dmc.Grid([
    #     dmc.Col([
    #         dash_table.DataTable(data=df_anime.to_dict('records'), page_size=12, style_table={'overflowX': 'auto'})
    #     ], span=6),
    #     dmc.Col([
    #         dcc.Graph(figure={}, id='graph-placeholder')
    #     ], span=6),
    # ]),

#     html.Div([
#     html.Button("Add Filter", id="redundant", n_clicks=0),
#     html.Div(id='dynamic-dropdown-container-div', children=[]),
#     html.Div(id="dropdown-container-output-div"),   
#    # html.Div(children=output, id ="dynamic_image") 
# ]), 
    ],
  fluid=True,
)])

# @app.callback(
#     Output('dynamic-dropdown-container-div', 'children'),
#     Input('redundant', 'n_clicks')
#     )
# def display_dropdowns(n_clicks):
#     patched_children = Patch()

#     new_element = html.Div([
#         dcc.Dropdown(
#             ['NYC', 'MTL', 'LA', 'TOKYO'],
#             id={
#                 'type': 'city-dynamic-dropdown',
#                 'index': n_clicks
#             }
#         ),
#         html.Div(
#             id={
#                 'type': 'city-dynamic-output',
#                 'index': n_clicks
#             }
#         )
#     ])
#     patched_children.append(new_element)
#     return patched_children


# @app.callback(
#     Output({'type': 'city-dynamic-output', 'index': MATCH}, 'children'),
#     Input({'type': 'city-dynamic-dropdown', 'index': MATCH}, 'value'),
#     State({'type': 'city-dynamic-dropdown', 'index': MATCH}, 'id'),
# )
# def display_output(value, id):
#     return html.Div(f"Dropdown {id['index']} = {value}")


@app.callback(
    [Output('dynamic_image_1', 'children'),
    Output('dynamic_image_2', 'children'),
    Output('dynamic_image_0', 'children'),
    State('dynamic-add-filter-btn1', 'n_clicks')],
    Input('dynamic-add-filter-btn1', 'n_clicks'),
    Input('user_type', 'value'),
    State('choose', 'value'),    
    )
def display_images(state, n_clicks, user_type, dropdown):
    # patched_children1 = Patch()     
    # patched_children2 = Patch()  
    # patched_children3 = Patch()  
    print(dropdown)

    patched_children1 = []
    patched_children2 = []
    patched_children3 = []
    
    if state%2 !=0:
       # print(n_clicks)
        patched_children1 = []
        patched_children2 = []
        patched_children3 = []
        rec = Recommender()
        # fit recommender
       # rec.fit(rating_pth='data/rating.csv', content_pth= 'data/anime.csv', learning_rate=.002, iters=1)
        # predict
        # rec.predict_rating(user_id=8, movie_id=2844)
        # # make recommendations
        # print(rec.make_recommendations(8,'user')) # user in the dataset
        # print(rec.make_recommendations(1,'user')) # user not in dataset
        # print(rec.make_recommendations(1853728)) # movie in the dataset
        # print(rec.make_recommendations(1)) # movie not in dataset

        with open("recommender.pkl", "rb") as f:
            rec = pickle.load(f)

        popular_list = rec.make_recommendations(8, user_type, 12, dropdown)
        # print(rec.n_users)
        # print(rec.n_movies)
        # print(rec.num_ratings)


        output1 = []
     #   output2 = []
     #   output3 = []
        photo_id_list = [] 
        column = None
        for i in range(len( popular_list)):
            name =  textwrap.shorten(popular_list[i], width=30, placeholder='...')
            photo_id = random.randint(1,42)
            while photo_id in photo_id_list:
                photo_id = random.randint(1,42)
            photo_id_list.append(photo_id) 
            #output1.append(dbc.Col( [html.Img(src= 'https://raw.githubusercontent.com/michaelbabyn/plot_data/master/naphthalene.png' , style={'height':'80%',"width": "70%","text-align":"center"})
            column = dbc.Col( [html.Img(src= f"assets/{photo_id}.jpg" , style={'height':'80%',"width": "70%","text-align":"center"})           
            , html.Div(html.P(f'{name}', style ={"text-align":"left", "font-size":"14px","color":"black"} ),style ={"height":"20"})], style={'height':'10%',  "float": "left", "width": "25%","padding": "1px"}, id = {"type":"dynamic_image_1", "index" : n_clicks})
            if i-1 % 4 == 0 and i == 0 :
                column = dbc.Row(column, style ={"height":"100%"})
            output1.append(column)
          #  print(photo_id)
        patched_children1 = output1
    #    patched_children2.extend(output2)
      #  patched_children3.extend(output3)
        return patched_children1, patched_children2,patched_children3
    if state % 2 == 0:
        patched_children1 = []
        patched_children2 = []
        patched_children3 = []
        n_clicks += 1 
        return patched_children1, patched_children2,patched_children3
  #  print(n_clicks)

# @app.callback(
#     Output({'type': 'city-dynamic-output', 'index': MATCH}, 'children'),
#     Input({'type': 'city-dynamic-dropdown', 'index': MATCH}, 'value'),
#     State({'type': 'city-dynamic-dropdown', 'index': MATCH}, 'id'),
# )
# def display_anime_title(value, id):
#     return html.Div(f"Anime Name")


def make_table(dataframe):
    return dbc.Table.from_dataframe(
        dataframe, bordered=True, hover=True, responsive=True, striped=True, style={}
    )

try:  # wrapping this, since a forum post said it may be deprecated at some point.
    app.title = "A Mini Recommendation System"
except:
    print("Could not set the page title!")

# @app.callback(
#     Output('dynamic_image_3', 'children'),
#     Input('redundant', 'n_clicks'),
# )
# def clear_all(n_clicks):
#     output = Patch()  
#     if n_clicks == 1:
#         output = []
#     return output


# Run the App
if __name__ == '__main__':
    app.run_server(debug=True)