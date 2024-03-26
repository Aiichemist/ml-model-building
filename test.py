# 导入所需的库
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

# 设置应用标题
st.title('机器学习算法可视化工具')

# 生成随机数据集
st.sidebar.header('数据集设置')
n_samples = st.sidebar.number_input('样本数', min_value=100, max_value=10000, value=1000)
n_features = st.sidebar.number_input('特征数', min_value=2, max_value=100, value=20)
n_classes = st.sidebar.number_input('类别数', min_value=2, max_value=10, value=2)

X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=n_features//2, random_state=42)
df = pd.DataFrame(X, columns=[f'特征_{i+1}' for i in range(X.shape[1])])
df['目标'] = y

st.write('生成的数据集预览：')
st.write(df.head())

# 选择机器学习算法
st.sidebar.header('机器学习算法')
algorithm = st.sidebar.selectbox('选择算法', ['随机森林', '梯度提升树', '支持向量机'])

if algorithm == '随机森林':
    st.sidebar.subheader('随机森林参数')
    n_estimators = st.sidebar.number_input('树的数量', min_value=10, max_value=500, value=100)

    # 模型训练
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X, y)

    # 可视化
    st.subheader('随机森林模型结果')
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ConfusionMatrixDisplay.from_estimator(clf, X, y, ax=ax[0])
    RocCurveDisplay.from_estimator(clf, X, y, ax=ax[1])
    st.pyplot(fig)

    st.write(f'模型准确率：{accuracy_score(y, clf.predict(X)):.4f}')
    st.write("""
        随机森林是一种集成学习方法，通过构建多个决策树来提高模型的性能。它可以用于分类和回归任务，适用于各种数据集。
    """)

elif algorithm == '梯度提升树':
    st.sidebar.subheader('梯度提升树参数')
    learning_rate = st.sidebar.number_input('学习率', min_value=0.01, max_value=1.0, value=0.1)

    # 模型训练
    clf = GradientBoostingClassifier(learning_rate=learning_rate)
    clf.fit(X, y)

    # 可视化
    st.subheader('梯度提升树模型结果')
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot_confusion_matrix(clf, X, y, ax=ax[0])
    plot_roc_curve(clf, X, y, ax=ax[1])
    st.pyplot(fig)

    st.write(f'模型准确率：{accuracy_score(y, clf.predict(X)):.4f}')
    st.write("""
        梯度提升树是一种集成学习方法，通过迭代地训练决策树来改进模型性能。它以一种逐步改进的方式构建模型，通常在分类和回归问题中表现良好。
    """)

elif algorithm == '支持向量机':
    st.sidebar.subheader('支持向量机参数')
    kernel = st.sidebar.selectbox('核函数', ['linear', 'rbf', 'poly'])

    # 模型训练
    clf = SVC(kernel=kernel)
    clf.fit(X, y)

    # 可视化
    st.subheader('支持向量机模型结果')
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot_confusion_matrix(clf, X, y, ax=ax[0])
    plot_roc_curve(clf, X, y, ax=ax[1])
    st.pyplot(fig)

    st.write(f'模型准确率：{accuracy_score(y, clf.predict(X)):.4f}')
    st.write("""
        支持向量机是一种用于分类和回归分析的监督学习模型。它通过在特征空间中构建一个或多个超平面来进行分类或回归。
    """)
