# Covid 19 Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class covid_Analysis:
    def __init__(self, data_set="COVID19_Data_Analysis_Dataset.csv"):
        self.file_path = data_set
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Successfully loaded dataset: '{self.file_path}'\n")
        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found. Please check the path.")
            self.df = pd.DataFrame()
            return

        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'], dayfirst=True)
            self.df = self.df.sort_values(by='Date')

    def summary_statistics(self):
        print("\n=-=-=-=-=-=-=-=-=-= Dataset Overviwe -=-=-=-=-=-=-=-=-=-=")
        print(self.df.info())
        
        new_cases_arr = self.df['New_Cases'].to_numpy()
        
        print("\n=-=-=-=-=--=-=-=-=-= Statastical Summry -=-=-=--=-=-=-=-=-=-=")
        print(f"Mean Daily Cases:   {np.mean(new_cases_arr):,.2f}")
        print(f"Median Daily Cases: {np.median(new_cases_arr):,.2f}")
        print(f"Std Deviation:      {np.std(new_cases_arr):,.2f}")
        print(f"Max Daily Cases:    {np.max(new_cases_arr):,}")
        print(f"Min Daily Cases:    {np.min(new_cases_arr):,}")
        print("=============================================================")

    def add_numpy_features(self, window_size=7):
        """Computes rolling averages and log-transformed values via NumPy."""
        self.df['Log_New_Cases'] = np.log1p(self.df['New_Cases'].to_numpy())
        self.df['New_Cases_7Day_Avg'] = (
            self.df.groupby('Country')['New_Cases']
            .transform(lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='same'))
        )

    def plot_global_trends(self):
        print("\nGenerating Global COVID-19 Trends plot...")
        time_series = self.df.groupby('Date')[['Cumulative_Cases', 'Cumulative_Recovered', 'Cumulative_Deaths']].sum().reset_index()

        plt.figure(figsize=(12, 6))
        sns.set_theme(style="whitegrid")
        
        plt.plot(time_series['Date'], time_series['Cumulative_Cases'], label='Cases', color='blue', linewidth=2)
        plt.plot(time_series['Date'], time_series['Cumulative_Recovered'], label='Recoveries', color='green', linewidth=2)
        plt.plot(time_series['Date'], time_series['Cumulative_Deaths'], label='Deaths', color='red', linewidth=2)

        plt.title('Global COVID-19 Trends Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Metrics')
        plt.tight_layout()
        plt.show()

    def plot_country_comparison(self, top_n=5):
        print(f"\nGenerating Top {top_n} Country Comparison bar plot...")
        latest_data = self.df.groupby('Country')[['Cumulative_Cases', 'Cumulative_Recovered', 'Cumulative_Deaths']].max().reset_index()
        top_countries = latest_data.sort_values(by='Cumulative_Cases', ascending=False).head(top_n)

        plt.figure(figsize=(12, 6))
        top_countries_melted = top_countries.melt(id_vars='Country', var_name='Metric', value_name='Count')
        sns.barplot(data=top_countries_melted, x='Country', y='Count', hue='Metric', palette='viridis')

        plt.title(f'Top {top_n} Countries by COVID-19 Metrics', fontsize=14, fontweight='bold')
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Total Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_intervention_impact(self):
        print("\nGenerating Government Intervention Impact scatter plot...")
        self.add_numpy_features()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.df, 
            x='Government_Intervention_Score', 
            y='Log_New_Cases', 
            hue='Region', 
            alpha=0.7
        )
        
        
        x = self.df['Government_Intervention_Score'].to_numpy()
        y = self.df['Log_New_Cases'].to_numpy()
        mask = ~np.isnan(x) & ~np.isnan(y)
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        
        x_trend = np.linspace(x[mask].min(), x[mask].max(), 100)
        y_trend = slope * x_trend + intercept
        
        plt.plot(x_trend, y_trend, color='black', linestyle='--', label=f'Trend Line (Slope: {slope:.3f})')

        plt.title('Impact of Government Interventions on Log(New Cases)', fontsize=14, fontweight='bold')
        plt.xlabel('Government Intervention Score', fontsize=12)
        plt.ylabel('Log(New Cases + 1)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_country_trends(self):
        print("\nGenerating Cumulative Cases per Country plot...")
        latest_data = self.df.groupby('Country')['Cumulative_Cases'].max()
        top_countries = latest_data.nlargest(5).index.tolist()

        filtered_df = self.df[self.df['Country'].isin(top_countries)]

        plt.figure(figsize=(12, 6))
        for country in top_countries:
            c_data = filtered_df[filtered_df['Country'] == country]
            plt.plot(c_data['Date'], c_data['Cumulative_Cases'], label=country, linewidth=2)

        plt.title('COVID-19 Cumulative Cases (Top 5 Countries)', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Cases', fontsize=12)
        plt.legend(title='Country')
        plt.tight_layout()
        plt.show()


analyzer = covid_Analysis("COVID19_Data_Analysis_Dataset.csv")

    
while True:
    print("\n" + "=!"*24)
    print("            COVID-19 DATA ANALYSIS MENU    ")
    print("=!"*24)
    print("1. View Statistical Summary (Pandas & NumPy)")
    print("2. Plot Global Trends Over Time")
    print("3. Compare Top Countries (Bar Chart)")
    print("4. Analyze Government Intervention Impact")
    print("5. Plot Cumulative Case Trends by Country")
    print("6. Run All Visualizations sequentially")
    print("7. Exit")
    print("=!"*24)

    choice = input("Enter your choice (1-7):")

    if choice == '1':
        analyzer.summary_statistics()
    elif choice == '2':
        analyzer.plot_global_trends()
    elif choice == '3':
        try:
            n = int(input("How many top countries to display? : ") or 5)
            analyzer.plot_country_comparison(top_n=n)
        except ValueError:
            print("Invalid input. Displaying top 5 by default.")
            analyzer.plot_country_comparison(top_n=5)
    elif choice == '4':       
        analyzer.plot_intervention_impact()
    elif choice == '5':
        analyzer.plot_country_trends()
    elif choice == '6':
        analyzer.summary_statistics()
        analyzer.plot_global_trends()
        analyzer.plot_country_comparison()
        analyzer.plot_intervention_impact()
        analyzer.plot_country_trends()
    elif choice == '7':
        print("\nExiting COVID-19 Analysis Program. Goodbye!")
        break
    else:
        print("\n[Invalid Choice] Please select a number between 1 and 7.")



