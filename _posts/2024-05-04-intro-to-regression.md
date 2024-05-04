---
layout: post
title:  "Bài 5 : Làm quen với Python và Scikit-learn cho mô hình hồi quy"
date:   2024-05-04 09:00:00 +0700
author: "Phạm Bình An"
categories: 
---

![Tóm tắt về hồi quy trong một trang giấy]({{site.url}}/assets/images/classic-ML-course/ml-regression.png)

> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

## Bắt đầu

Trong bốn bài học về hồi quy, bạn sẽ khám phá cách xây dựng mô hình hồi quy. Nhưng trước đó, hãy đảm bảo bạn có sẵn các công cụ phù hợp để bắt đầu!

Trong bài học này, bạn sẽ học cách:

- Định cấu hình máy tính của bạn cho các tác vụ học máy cục bộ.
- Làm việc với Jupyter notebook.
- Cài đặt và sử dụng Scikit-learn.
- Thực hành.


## Cài đặt và định cấu hình

[![ML for beginners - Setup your tools ready to build Machine Learning models](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML for beginners -Setup your tools ready to build Machine Learning models")

> 🎥 Click ảnh trên để xem video về cách định cấu hình để lập trình học máy.

1. **Cài Python** [tại đây](https://www.python.org/downloads/). Bạn sẽ sử dụng Python cho nhiều tác vụ khoa học dữ liệu và học máy. Hầu hết các máy tính đều đã cài đặt Python. 

   Đôi khi, bạn có thể cần các phiên bản Python khác nhau cho các dự án khác nhau. Khi đó, bạn nên dùng [môi trường ảo](https://docs.python.org/3/library/venv.html).

2. **Cài Visual Studio Code** [tại đây](https://code.visualstudio.com/). Sau đó hãy [thiết lập Visual Studio Code cho Python](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott).

   > Làm quen với Python qua [khóa học này](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Setup Python with Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Setup Python with Visual Studio Code")
   >
   > 🎥 Click ảnh trên để xem video hướng dẫn cách dùng Python trong Visual Studio Code

3. **Cài Scikit-learn** theo [hướng dẫn sau](https://scikit-learn.org/stable/install.html). 

1. **Cài Jupyter Notebook** [tại đây](https://pypi.org/project/jupyter/).

## Môi trường viết code học máy của bạn

Bạn sẽ sử dụng **notebook** để viết code Python và tạo các mô hình học máy. Loại tệp này là một công cụ phổ biến dành cho các nhà khoa học dữ liệu và chúng có đuôi file `.ipynb`.

Notebook là một môi trường tương tác cho phép nhà phát triển vừa viết mã, vừa thêm ghi chú cho mã, điều này khá hữu ích cho các dự án mang tính thử nghiệm hoặc định hướng nghiên cứu.

[![ML for beginners - Set up Jupyter Notebooks to start building regression models](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML for beginners - Set up Jupyter Notebooks to start building regression models")

> 🎥 Click ảnh trên để xem video bài học

### Thực hành - làm việc với notebook

1. Tạo file _notebook.ipynb_ và mở trong Visual Studio Code.

   Lúc này, một server Jupyter với Python 3+ sẽ khởi động.

1. Chọn biểu tượng `md` và thêm văn bản **# Welcome to your notebook**.

   Tiếp theo, hãy viết ít code nha

1. Gõ **print('hello notebook')** trong phần code.
1. Chọn biểu tượng mũi tên để chạy code

   Bạn sẽ thấy output sau

    ```output
    hello notebook
    ```

![VS Code with a notebook open]({{site.url}}/assets/images/classic-ML-course/notebook.jpg)

## Làm việc với Scikit-learn
Bây giờ bạn đã có Python và đã quen thuộc với Jupyter notebooks, hãy bắt đầu làm quen với Scikit-learn (phát âm như `sai-kit lơn`). Scikit-learn cũng cung cấp [API mở rộng](https://scikit-learn.org/stable/modules/classes.html#api-ref) để giúp bạn thực hiện các tác vụ học máy.

Theo [trang web](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn là một thư viện máy học nguồn mở hỗ trợ việc học có giám sát và không giám sát. Nó cũng cung cấp nhiều công cụ khác nhau để điều chỉnh mô hình, tiền xử lý dữ liệu, lựa chọn và đánh giá mô hình và nhiều tiện ích khác."

## Thực hành - notebook đầu tiên

> Bài học này dựa trên [các ví dụ hồi quy tuyến tính](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) được cung cấp trên website của Scikit-learn


[![ML for beginners - Your First Linear Regression Project in Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML for beginners - Your First Linear Regression Project in Python")

> 🎥 Click ảnh trên để xem video bài học

Trong phần này, bạn sẽ làm việc với một tập dữ liệu nhỏ về bệnh tiểu đường được tích hợp vào Scikit-learn cho mục đích học tập. Hãy tưởng tượng rằng bạn muốn thử nghiệm một phương pháp điều trị cho bệnh nhân tiểu đường. Các mô hình Machine Learning có thể giúp bạn xác định bệnh nhân nào sẽ đáp ứng tốt hơn với phương pháp điều trị, dựa trên sự kết hợp của các biến số. Ngay cả một mô hình hồi quy rất cơ bản, khi được hình dung, có thể hiển thị thông tin về các biến số có thể giúp bạn tổ chức các thử nghiệm lâm sàng lý thuyết của mình.

✅ Có nhiều loại phương pháp hồi quy và việc bạn chọn loại nào tùy thuộc vào câu trả lời bạn đang tìm kiếm. Nếu bạn muốn dự đoán chiều cao có thể xảy ra của một người ở một độ tuổi nhất định, bạn nên sử dụng hồi quy tuyến tính vì bạn đang tìm kiếm **giá trị số**. Nếu bạn muốn khám phá xem liệu một loại món ăn có nên được coi là thuần chay hay không thì bạn đang tìm kiếm **bài toán phân lớp** với mô hình hồi quy logistic. Bạn sẽ tìm hiểu thêm về hồi quy logistic sau. 

### Nhập các thư viện

Dưới đây là một vài thư viện chúng ta cần cho tác vụ này:

- **matplotlib**. Đây là một [công cụ vẽ đồ thị](https://matplotlib.org/) hữu ích mà chúng ta sẽ sử dụng để tạo một biểu đồ đường.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) là một thư viện hữu ích để xử lý dữ liệu số trong Python.
- **sklearn**. Đây chính là tên khai báo của thư viện [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

1. Nhập các thư viện bằng các dòng code sau

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Ở trên bạn đã khai báo các thư viện `matplotlib`, `numpy` và bạn cũng nhập `datasets`, `linear_model` and `model_selection` từ thư viện `sklearn`. `model_selection` được dùng để chia tập dữ liệu thành tập huấn luyện và tập thử nghiệm.

### Bộ dữ liệu

[Bộ dữ liệu tiểu đường](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) có 442 mẫu dữ liệu xoay quanh tiểu đường, với 10 biến đặc trưng, bao gồm:

- Tuổi: đơn vị là _năm_
- bmi: Chỉ số khối cơ thể
- bp: Huyết áp trung bình
- s1 tc: Tế bào T (một loại tế bào bạch cầu)

✅ Tập dữ liệu này bao gồm khái niệm 'giới tính' như một biến đặc trưng quan trọng để nghiên cứu về bệnh tiểu đường. Nhiều bộ dữ liệu y tế bao gồm cách phân loại nhị phân này. Cách phân loại này có thể loại trừ một bộ phận dân số khỏi các phương pháp điều trị

Bây giờ, hãy tải dữ liệu X và y lên.

> 🎓 Hãy nhớ rằng, đây là học có giám sát và chúng ta cần một mục tiêu gọi là 'y'.

Trong một ô code mới, hãy tải tập dữ liệu về bệnh tiểu đường bằng cách gọi `load_diabetes()`. Đầu vào `return_X_y=True` báo hiệu rằng `X` sẽ là ma trận dữ liệu và `y` sẽ là mục tiêu hồi quy.

1. Thêm một số lệnh print để hiển thị hình dạng của ma trận dữ liệu và phần tử đầu tiên của nó:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```



    Bạn có thể thấy tập dữ liệu này có 442 mẫu với 10 đặc trưng

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

2. Tiếp theo, hãy chọn một phần của tập dữ liệu này để vẽ biểu đồ bằng cách chọn cột thứ 3 của tập dữ liệu. Bạn có thể làm điều này bằng cách sử dụng toán tử `:` để chọn tất cả các hàng, sau đó chọn cột thứ 3 bằng cách sử dụng chỉ mục (2). Bạn cũng có thể định hình lại dữ liệu thành một mảng 2D - như yêu cầu cho việc vẽ biểu đồ - bằng cách sử dụng `reshape(n_rows, n_columns)`. Nếu một trong các tham số là -1, thì kích thước tương ứng sẽ được tính toán tự động.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ In dữ liệu ra để kiểm tra hình dạng của.

3. Bây giờ bạn đã có dữ liệu sẵn sàng để vẽ, bạn có thể xem liệu máy có thể giúp xác định sự phân chia logic giữa các số trong tập dữ liệu này hay không. Để làm điều này, bạn cần chia cả dữ liệu (X) và mục tiêu (y) thành các tập huấn luyện và thử nghiệm. Đoạn code dưới đây sẽ tạo tập kiểm thử từ 33% bộ dữ liệu gốc

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Bây giờ hãy bắt đầu huấn luyện mô hình bằng cách tải mô hình hồi quy tuyến tính lên và huấn luyện bằng `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ Bạn có thể tìm thấy hàm `model.fit()` trong nhiều thư viện học máy như TensorFlow

5. Bây giờ, ta sẽ cho mô hình dự đoán trên tập thử nghiệm với hàm `predict()`. 

    ```python
    y_pred = model.predict(X_test)
    ```

6. Bây giờ, hãy vẽ một biểu đồ hiển thị tập dữ liệu thử nghiệm. Matplotlib là một công cụ rất hữu ích cho nhiệm vụ này. Tạo một biểu đồ phân tán của tất cả dữ liệu thử nghiệm X và y và sử dụng kết quả dự đoán của mô hình để vẽ một đường thẳng ở vị trí thích hợp nhất, giữa các nhóm dữ liệu của mô hình.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![a scatterplot showing datapoints around diabetes]({{site.url}}/assets/images/classic-ML-course/scatterplot.png)

   ✅ Bây giờ bạn hãy thử đoán ý nghĩa của đường thẳng này nhé. Đáp án sẽ có trong bài 7

Xin chúc mừng, bạn đã xây dựng mô hình hồi quy tuyến tính đầu tiên của mình, dùng mô hình để dự đoán và vẽ biểu đồ! Các lý thuyết về mô hình này sẽ được giải đáp trong bài 7

---
