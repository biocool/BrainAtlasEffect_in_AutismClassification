import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots


def categorize_age(age):
    """Categorize age into specified groups."""
    if 6 <= age <= 10:
        return '6-10'
    elif 10 < age <= 20:
        return '10-20'
    elif 20 < age <= 50:
        return '20-40'
    elif age > 50:
        return '>50'


if __name__ == '__main__':
    # Load data from the Excel file
    df = pd.read_excel('../Labels.csv')  # Ensure the correct path to the file
    df = df.loc[(~ df['Age'].isna()) & (~ df['Sex'].isna())]

    # Categorize the ages
    df['Age Group'] = df['Age'].apply(categorize_age)

    # Calculate counts for each combination of Group and SITE_ID
    grouped_df_sub1 = df.groupby(['Group', 'Sex', 'Age Group']).size().reset_index(name='Count')
    grouped_df_sub2 = df.groupby(['Group', 'SITE_ID']).size().reset_index(name='Count')

    # Calculate the total count to determine percentages
    total_count_sub1 = grouped_df_sub1['Count'].sum()
    total_count_sub2 = grouped_df_sub2['Count'].sum()

    # Calculate overall percentages for the main groups
    overall_percentages_sub1 = grouped_df_sub1.groupby('Group')['Count'].sum() / total_count_sub1 * 100
    overall_percentages_sub1 = overall_percentages_sub1.reset_index(name='Overall_Percentage')

    overall_percentages_sub2 = grouped_df_sub2.groupby('Group')['Count'].sum() / total_count_sub2 * 100
    overall_percentages_sub2 = overall_percentages_sub2.reset_index(name='Overall_Percentage')

    # Merge overall percentages into the grouped data
    grouped_df_sub1 = grouped_df_sub1.merge(overall_percentages_sub1, on='Group')
    grouped_df_sub2 = grouped_df_sub2.merge(overall_percentages_sub2, on='Group')

    # Create the sunburst plot
    fig1 = px.sunburst(
        grouped_df_sub1,
        path=['Group', 'Sex', 'Age Group'],  # Define the hierarchy
        values='Count',  # Use the computed count for sizing
        color='Group',  # Color the plot by 'Group' to differentiate
        color_discrete_map={'Control': 'blue', 'Autism': 'red'},  # Color mapping
    )

    fig2 = px.sunburst(
        grouped_df_sub2,
        path=['Group', 'SITE_ID'],  # Define the hierarchy
        values='Count',  # Use the computed count for sizing
        color='Group',  # Color the plot by 'Group' to differentiate
        color_discrete_map={'Control': 'blue', 'Autism': 'red'},  # Color mapping
    )

    # Add text information directly to each segment
    fig1.update_traces(
        textinfo='label+percent entry',
    )

    fig2.update_traces(
        textinfo='label+percent entry',  # Show both the label and percentage on each segment
    )

    # Update layout for better readability
    fig1.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        sunburstcolorway=["#636EFA", "#EF553B"],  # Custom colors
    )

    fig2.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        sunburstcolorway=["#636EFA", "#EF553B"],  # Custom colors
    )

    # Create a subplot figure with two sunburst charts
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'domain'}]],
    )

    # Add both figures to the subplot
    fig.add_trace(fig1.data[0], row=1, col=1)
    fig.add_trace(fig2.data[0], row=1, col=2)

    # Update layout for better readability
    fig.update_layout(
        sunburstcolorway=["#636EFA", "#EF553B"]  # Custom colors
    )

    # Save the plot as a PDF
    fig.write_image("Fig1.pdf", format='pdf', engine="kaleido")
