# coffee-quality-project

# ☕ Coffee Quality Risk Classification Project

## 📌 Overview

This project develops a machine learning pipeline to classify coffee quality as **Good (0)** or **Defective (1)** based on sensory and physical attributes.

The solution is designed to support **coffee quality control systems** such as those used by URKU, helping detect defective coffee early and improve storage and processing decisions.

---

## 🎯 Objectives

* Build a supervised machine learning model to classify coffee quality
* Identify key factors influencing coffee defects
* Provide actionable insights for improving coffee storage conditions
* Support real-world decision-making in coffee production

---

## 🧠 Machine Learning Approach

### ✔ Problem Type

* **Binary Classification**

  * 0 = Good Coffee
  * 1 = Defective Coffee

### ✔ Target Logic

A coffee sample is labelled as **Defective** if:

* Category One Defects > 0 OR
* Category Two Defects > 0

---

## 🤖 Models Implemented

### 1. Logistic Regression

* Baseline model
* Requires feature scaling
* Interpretable results

### 2. Random Forest (Main Model)

* Handles non-linear relationships
* More robust performance
* Provides feature importance

---