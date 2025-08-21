# Basket Option Monte Carlo Simulation and Variance Reduction methods

## Overview
This project implements Monte Carlo simulations for pricing basket options with control variates techniques. 
It demonstrates variance reduction using the geometric basket, Curran (1994) and Yongchao Sun, Chenglong Xu (2018) methods. 
The goal is to study the efficiency of different estimators under realistic and datasets and provide some limitations. We study 4 methods of variance reduction based on control variate.



## Key Results

- Variance reduction methods' effectiveness varies strongly depending on basket composition and strike price.  
- They are highly effective under theoretical conditions, but can be limited with real datasets containing heterogeneous asset prices. Under the theoretical framework, the effort ratio reaches up to $1.30e+04$ and the variance reduction up to $4.91e+05$. 
- Increasing the variance of the underlying variable (e.g., by increasing the time horizon) reduces the effectiveness of variance reduction.  
- For real datasets with relatively homogeneous prices, the reduction works very well: the effort ratio reaches up to $1.14e+03$ and the variance reduction up to $2.16e+04$.



## Project Structure

- `notebooks/`  
  Jupyter notebooks with simulations and analysis:  
  - `variance_reduction_theory.ipynb` – theoretical exploration of variance reduction methods  
  - `monte_carlo_reduction_variance_real_data.ipynb` – simulations using a real dataset

- `scripts/`  
  Python scripts containing functions used in the notebooks:  
  - `Variance_reduction.py`

- `data/`  
  Dataset used in the notebooks:  
  - `prices.xlsx`

- `report/`  
  PDF report summarizing results:  
  - `Variance_reduction.pdf`



