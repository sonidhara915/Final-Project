# Titanic Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(filepath='Titanic-Dataset.csv'):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"\nError: Could not find '{filepath}'. Ensure the file is in the current directory.")
        return None

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    return df

def show_summary_stats(df):
    print("\n","=!"*20)
    print("         Dataset Summury ")
    print("=!"*20)
    print(f"Total Passengers Analyzed : {len(df)}")
    print(f"Overall Survival Rate     : {df['Survived'].mean() * 100:.2f}%\n")
    
    print(" Survival Rate by Sex ")
    print((df.groupby('Sex')['Survived'].mean() * 100).map('{:.2f}%'.format))
    
    print("\n Survival Rate(%) By Class ")
    print((df.groupby('Pclass')['Survived'].mean() * 100).map('{:.2f}%'.format))

    print("\n Survival Rate by Class & Sex ")
    print((df.groupby(['Pclass', 'Sex'])['Survived'].mean() * 100).map('{:.2f}%'.format))
    print("="*40)

def show_data_overview(df):

    print("\n","*40")
    print("            DATA OVERVIEW")
    print("="*40)
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values Count:")
    print(df.isnull().sum())
    print("="*40)

def plot_survival_by_gender(df):
    """Plots survival rate by gender."""
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x='Sex', y='Survived', palette='Set1')
    plt.title('Survival Rate by Gender')
    plt.ylabel('Survival Rate')
    plt.show()

def plot_survival_by_class(df):
    """Plots survival rate by ticket class and gender."""
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex', palette='Set1')
    plt.title('Survival Rate by Passenger Class & Gender')
    plt.ylabel('Survival Rate')
    plt.xlabel('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)')
    plt.show()

def plot_age_distribution(df):
    """Plots age distribution separated by survival status."""
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='Age', hue='Survived', kde=True, element='step', palette='Set2')
    plt.title('Age Distribution by Survival Status')
    plt.xlabel('Age')
    plt.ylabel('Passenger Count')
    plt.show()

def plot_full_dashboard(df):
    """Generates a complete 2x2 multi-panel visualization board."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    
    sns.countplot(ax=axes[0, 0], data=df, x='Survived', palette='Set2')
    axes[0, 0].set_title('Overall Survival Count')
    axes[0, 0].set_xticklabels(['Died', 'Survived'])

    
    sns.barplot(ax=axes[0, 1], data=df, x='Sex', y='Survived', palette='Set1')
    axes[0, 1].set_title('Survival Rate by Gender')

    
    sns.barplot(ax=axes[1, 0], data=df, x='Pclass', y='Survived', hue='Sex', palette='Set1')
    axes[1, 0].set_title('Survival Rate by Class & Gender')

   
    sns.histplot(ax=axes[1, 1], data=df, x='Age', hue='Survived', kde=True, element='step', palette='Set2')
    axes[1, 1].set_title('Age Distribution by Survival Status')

    plt.tight_layout()
    plt.show()

while True:
    
    df = load_and_clean_data()
    
    if df is None:
        print("Exiting application due to missing dataset file.")
        break


    print("\n","=!"*20)
    print("    TITANIC SURVIVAL ANALYSIS MENU")
    print("=!"*20)
    print("1. View Dataset Overview & Data Info")
    print("2. View Statistical Survival Rates")
    print("3. Plot Survival Rate by Gender")
    print("4. Plot Survival Rate by Passenger Class & Gender")
    print("5. Plot Age Distribution by Survival Status")
    print("6. Display Full Visualization Dashboard")
    print("7. Exit")
    print("="*40)

    choice = input("Enter your option (1-7): ").strip()

    if choice == '1':
        show_data_overview(df)
    elif choice == '2':
        show_summary_stats(df)
    elif choice == '3':
        plot_survival_by_gender(df)
    elif choice == '4':
            plot_survival_by_class(df)
    elif choice == '5':
        plot_age_distribution(df)
    elif choice == '6':
        plot_full_dashboard(df)
    elif choice == '7':
        print("\nExiting Titanic Analysis Tool. Goodbye!")
        break
    else:
        print("\nInvalid choice. Please select a number between 1 and 7.")
