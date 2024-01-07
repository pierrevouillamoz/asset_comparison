# asset_comparison

How to compare asset to invest in ? What assets buy or sell, when and how much ?

When I decided to learn about finance the first step was finding good material for my education.

I found that there are 3 paradigms that can be used together :
- technical analysis
- fundamental analysis
- quantitative analysis

As I have a good background in mathematics, I prefer the third way. 
This one is perfect with python tool and once computation is made, you only need to check results on a regular basis (monthly or annualy).

## my code

In 2022 I started the MOOC Investment Management with Python and Machine Learning available in coursera.org
https://www.coursera.org/specializations/investment-management-python-machine-learning

I only did the half of the first module: Markowitz optimization. However the content of the MOOC is great and knowledge learnt enough to try investment strategy.

With the tools of the edhec risk kit module, I develop a backtest function :
- A for loop runs trough time serie of asset value
- At at first moment, afetr X years, initial portfolio allocations are computed using an optimizer over the X first years of time series
- Each month portfolio values are updated
- Each year portfolio are rebalanced through the initial allocation or rebalanced through a new allocation based on the X lasted years

Inputs of the function are :
- dataset of time series
- initial amount to invest
- an amount to invest each month called DCA (dollar cost average is the name of this investment style)
- desired gross rate
- risk free asset rate
- a delay before investing (in year)
- asset name to make comparison upon invest in this one

Several output are computed because our function make different portfolio to make better comparison :
- Portfolio value with a rolling optimization each year
- Portfolio composition trough year
- Portfolio value with a fixed allocation computed once a time
- Portfolio composition
- Value of an invest on risk free asset (DCAmin)
- Value of an invest on a risk free asset with the desired gross rate (DCAmax - utopical)
- Value of an invest based on one specified asset (DCAasset)
- Total of amount invested


