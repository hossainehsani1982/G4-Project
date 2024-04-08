import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pywaffle import Waffle

import os


dfs = []
combined_dataset_path = r'E:\NpowerLab\DataSets\Project\combined_dataset.csv'

if os.path.isfile(combined_dataset_path):
    pass
else:
    for fileName in os.listdir(r'E:\NpowerLab\DataSets\Project'):
         if fileName.endswith('.csv'):
            df = pd.read_csv(os.path.join(r'E:\NpowerLab\DataSets\Project', fileName))
            df = df.iloc[:, :12]
            dfs.append(df)
            
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(combined_dataset_path, index=False)


df = pd.read_csv(combined_dataset_path)

'''
Data Cleaning
''' 

#remove rows without date
df = df.dropna(subset=['Date'])

#remove rows with invalid date
date_pattern = r"\d{4}-\d{2}-\d{2}"
df = df[df['Date'].str.match(date_pattern)]

#remove rows with nan values
nan_columns = df.columns[df.isna().any()].tolist()
df.drop(nan_columns, axis=1, inplace=True)

'''
Data Transformation
'''

#convert date to datetime
df['Date'] = pd.to_datetime(df['Date'])
#replace date with year
df['Date'] = df['Date'].dt.year
#rename date column to year
df.rename(columns={'Date': 'Year'}, inplace=True)

print(df[['Year','Symbol', 'Open', 'Close', 'Volume']].head(100))   


####plotting
#plotting bar chart

df = df.loc[:, ['Year', 'Symbol', 'Open', 'Close', 'Volume']]
df = df.groupby(['Year', 'Symbol']).sum()
print(df.head(20))
df.reset_index(inplace=True)

desired_year = 2001  # Change this to the desired year

# Filter data for the desired year
df_year = df[df['Year'] == desired_year]

# Create a pivot table to reshape the data
pivot_df = df_year.pivot_table(index='Symbol', columns='Year', values=['Open', 'Close'])

# Plot column chart
fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.35
symbols = pivot_df.index
index = range(len(symbols))

open_prices = pivot_df[('Open', desired_year)].values
close_prices = pivot_df[('Close', desired_year)].values

ax.bar(index, open_prices, bar_width, label='Open', zorder=3)
ax.bar([x + bar_width for x in index], close_prices, bar_width, label='Close', zorder=3)

ax.set_xlabel('Symbol')
ax.set_ylabel('Price')
ax.set_title(f'Open and Close Prices for Symbols in {desired_year}')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(symbols, rotation=45, ha='right') 
ax.set_xticklabels(symbols)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
plt.tight_layout()
plt.show()



####plotting
#plotting waffle chart

df.set_index('Symbol', inplace=True)
waffle_df = df.loc[:, ['Open', 'Close', 'Volume']]
waffle_df = waffle_df.groupby('Symbol').sum()

unique_symbols = waffle_df.index.unique()
num_symbols = len(unique_symbols)

colormaps = ['tab20', 'tab20b', 'tab20c']  
colors_per_color_map = int(np.ceil(65 / len(colormaps))) # 65/3 = 21.6666

colors = []
for cmap in colormaps:
    print(cmap)
    colormap = plt.colormaps.get_cmap(cmap)
    # Generate equally spaced positions along the colormap
    positions = np.linspace(0, 1, colors_per_color_map)
    print(f'positions : {positions}')
    # Sample colors from the colormap at those positions
    sampled_colors = colormap(positions)
    print(sampled_colors)
    # Add the sampled colors to the colors list
    colors.extend(sampled_colors)


# Take only the required number of colors
colors = colors[:65]

color_dict = {}
for i, symbol in enumerate(unique_symbols):
    color_dict[symbol] = colors[i]


fig = plt.figure(
    FigureClass= Waffle,
    rows=30,
    columns =60,
    values=waffle_df['Volume'],
    colors=[color_dict[symbol] for symbol in waffle_df.index]
)

legend_handles = [
    plt.Line2D([0], [0], marker='o', color=color_dict[symbol], label=symbol, markersize=10)
    for symbol in waffle_df.index.unique()
]

# Add legend with handles and labels
plt.legend(handles=legend_handles, loc='lower right', bbox_to_anchor=(0, -0.1), title='Symbols', ncol=3)

plt.show()

####plotting
#plotting regression plot

df.reset_index(inplace=True)
#df.set_index('Year', inplace=True)
df = df.loc[:, ['Year', 'Symbol', 'Open', 'Close', 'Volume']]
df = df.groupby(['Year', 'Symbol']).sum()
print(df.head(20))
open_colose_mean = df[['Open', 'Close']].mean(axis = 1)
df['Total'] = (df['Volume'] * open_colose_mean)/df['Volume'].sum()
df.reset_index(inplace=True)

sns.regplot(x= 'Year', y='Total', data=df)


####plotting
#plotting area plot of the stock price of the companies over the years

df = df.pivot(index='Year', columns='Symbol', values='Total')
ax = df.plot.area()
years = df.index
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45)

formatter = plt.FuncFormatter(lambda x, _: "${:,.0f}".format(x))
ax.yaxis.set_major_formatter(formatter)


plt.title('Stock Price of Companies Over the Years')
plt.ylabel('Stock Price')
plt.xlabel('Years')
plt.xlim(df.index.min(), df.index.max())
plt.legend(title='Companies', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()



# Dahsboard

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go

# Ensure the correct column names
combined_dataset_path = r'E:\NpowerLab\DataSets\Project\combined_dataset.csv'

df1 = pd.read_csv(combined_dataset_path)

df1 = df1.dropna(subset=['Date'])

#remove rows with invalid date
date_pattern = r"\d{4}-\d{2}-\d{2}"
df1 = df1[df1['Date'].str.match(date_pattern)]

#remove rows with nan values
nan_columns = df1.columns[df1.isna().any()].tolist()
df1.drop(nan_columns, axis=1, inplace=True)

'''
Data Transformation
'''

#convert date to datetime
df1['Date'] = pd.to_datetime(df1['Date'])
#replace date with year
df1['Date'] = df1['Date'].dt.year
#rename date column to year
df1.rename(columns={'Date': 'Year'}, inplace=True)


print(df1.head())
print(df1.columns)

# Grouping and summing the data
df1 = df1[['Year', 'Symbol', 'Open', 'Close', 'Volume']].groupby(['Year', 'Symbol']).sum().reset_index()

# Set the default year
desired_year = 2001  # Change this to the desired year

# Filter data for the default year
df_year = df1[df1['Year'] == desired_year]

# Create a pivot table to reshape the data
pivot_df = df_year.pivot_table(index='Symbol', columns='Year', values=['Open', 'Close'])

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Open and Close Prices Dashboard"),
    html.Div([
        html.Label("Select Year:"),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(year), 'value': year} for year in df1['Year'].unique()],
            value=desired_year,
            placeholder="Select a year"
        )
    ]),
    dcc.Graph(id='price-chart')
])

# Define the callback to update the chart based on the selected year
@app.callback(
    Output('price-chart', 'figure'),
    [Input('year-dropdown', 'value')]
)
def update_chart(selected_year):
    # Filter data for the selected year
    df_year = df1[df1['Year'] == selected_year]
    
    # Create a pivot table to reshape the data
    pivot_df = df_year.pivot_table(index='Symbol', columns='Year', values=['Open', 'Close'])
    
    # Create traces for the bar chart
    traces = []
    for column in pivot_df.columns.levels[1]:
        traces.append(go.Bar(
            x=pivot_df.index,
            y=pivot_df[('Open', column)],
            name=f'{column} - Open'
        ))
        traces.append(go.Bar(
            x=pivot_df.index,
            y=pivot_df[('Close', column)],
            name=f'{column} - Close'
        ))

    # Set layout for the bar chart
    layout = go.Layout(
        barmode='group',
        xaxis=dict(tickangle=45, tickmode='array', tickvals=list(range(len(pivot_df.index))), ticktext=pivot_df.index)
    )

    # Create the figure
    figure = go.Figure(data=traces, layout=layout)

    return figure

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
