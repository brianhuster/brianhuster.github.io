---
layout: post
title:  "Bài 6 : Trực quan hóa dữ liệu cho mô hình hồi quy"
date:   2024-06-21 21:57:00 +0700
author: "Phạm Bình An"
categories: 
---

---

Xem thêm các bài học khác của khóa **học máy cơ bản** tại [đây]({{site.url}}/2024/04/30/classic-ML-course.html)

---

# Chuẩn bị và trực quan hóa dữ liệu

![Data visualization infographic](./images/data-visualization.png)

Infographic by [Dasani Madipalli](https://twitter.com/dasani_decoded)

## Giới thiệu

Trong bài này, bạn sẽ học cách chuẩn bị dữ liệu cho việc xây dựng mô hình và sử dụng Matplotlib để trực quan hóa dữ liệu

## Đặt câu hỏi về bộ dữ liệu

Câu hỏi sẽ quyết định thuật toán học máy bạn dùng. Và chất lượng câu trả lời phụ thuộc vào bản chất bộ dữ liệu

Hãy nhìn qua [bộ dữ liệu](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) được dùng trong bài học này. Bạn có thể mở file .csv trong VS code. Chỉ nhìn qua, bạn cũng có thể nhận ra bộ dữ liệu khá lộn xộn, có nhiều ô trống, hỗn hợp dữ liệu chữ và số,...

[![Học máy cơ bản : cách phân tích và làm sạch dữ liệu](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML for beginners - How to Analyze and Clean a Dataset")

> 🎥 Click vào ảnh để xem video về cách phân tích và làm sạch dữ liệu
Trên thực tế, không phải lúc nào bạn cũng nhận được bộ dữ liệu đủ tốt để đưa luôn vào mô hình học máy. Trong bài học này, bạn sẽ học cách chuẩn bị bộ dữ liệu sơ bộ bằng các thư viện Python chuẩn. Bạn cũng sẽ học cách trực quan hóa dữ liệu.

## Nghiên cứu trường hợp : thị trường bí ngô

In thư mục này bạn sẽ tìm thấy file .csv trong thư mục root `data` có tên là [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) gồm 1757 dòng dữ liệu về thị trường, sắp xếp theo thành phố. Đây là dữ liệu trích xuất từ [Báo cáo Tiêu chuẩn Thị trường Đầu mối về Cây trồng Chuyên biệt](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) cung cấp bởi Bộ Nông nghiệp Hoa Kỳ.

### Chuẩn bị dữ liệu

Bộ dữ liệu này thuộc về công chúng (public domain), nghĩa là tất cả mọi người đều có quyền tải xuống và sử dụng cho các mục đích. Thông thường, bạn sẽ phải tải xuống nhiều file, sẽ khá bất tiện, vì vậy, Microsoft đã gộp sẵn thành 1 file csv duy nhất
 
### Dữ liệu bí ngô - kết luận ban đầu

Bạn thấy gì từ bộ dữ liệu này? Có lẽ bạn đã nhận ra rằng có hỗn hợp chuỗi, số, và các giá trị trống. 

What question can you ask of this data, using a Regression technique? What about "Predict the price of a pumpkin for sale during a given month". Looking again at the data, there are some changes you need to make to create the data structure necessary for the task.

## Thực hành - phân tích dữ liệu bí ngô

Hãy dùng Panda [Pandas](https://pandas.pydata.org/), (cái tên `Panda` là viết tắt của `Python Data Analysis (phân tích dư liệu Python)`), một công cụ rất hữu ích cho việc định hình, phân tích và chuẩn bị dữ liệu bí ngô.

### Đầu tiên, kiểm tra các ngày tháng bị thiếu

Bạn sẽ thực hiện các bước sau để kiểm tra ngày tháng bị thiếu:

1. Chuyển đổi dữ liệu ngày tháng thành định dạng tháng (kiểu ngày tháng Mỹ, `MM/DD/YYYY`).
2. Trích tách tháng sang một cột khác

Mở _notebook.ipynb_ file trong Visual Studio Code và nhập bảng dữ liệu bí ngô vào.

1. Dùng hàm `head()` để xem 5 hàng đầu tiên.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Bạn sẽ dùng hàm nào để xem 5 hàng cuối?

1. Kiểm tra dữ liệu bị thiếu:

    ```python
    pumpkins.isnull().sum()
    ```

    Có những dữ liệu bị thiếu, nhưng có lẽ nó k:hông cần thiết cho nhiệm vụ hiện tại.

1. Để làm cho khung dữ liệu của bạn dễ dàng làm việc hơn, chỉ chọn các cột bạn cần, sử dụng hàm `loc` để trích xuất từ khung dữ liệu gốc một nhóm hàng (được truyền dưới dạng tham số đầu tiên) và các cột (được truyền dưới dạng tham số thứ hai). Toán tử `:` trong trường hợp bên dưới có nghĩa là "tất cả các hàng".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Xác định mức giá trung bình của bí ngô

Hãy nghĩ về cách xác định giá bí ngô trung bình trong một tháng nhất định. Bạn cần những cột nào cho bài toán này? Gợi ý : bạn cần 3 cột

Giải đáp: lấy mức trung bình của cột `Low price` (giá thấp) và `High price` (giá cao) để điền vào cột `Price` mới và chuyển đổi cột `Date` để chỉ hiển thị tháng. May mắn thay, theo kiểm tra ở trên, không có dữ liệu nào bị thiếu về ngày tháng hoặc giá cả.

1. Để tính mức giá trung bình và trích thuất tháng, thêm đoạn code sau:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Bạn có thể kiểm tra giá trị bằng cách dùng hàm `print()`.

2. Bây giờ, sao chép dữ liệu đã chuyển đổi của bạn vào khung dữ liệu Pandas mới:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    In ra khung dữ liệu sẽ hiển thị cho bạn một tập dữ liệu rõ ràng, gọn gàng mà trên đó bạn có thể xây dựng mô hình hồi quy mới của mình.

### Khoan đã! Có gì đó lạ lắm

Nếu bạn nhìn vào cột "Package", các quả bí ngô được bán với nhiều cấu hình khác nhau. Một số được bán theo đơn vị "1 1/9 giạ (bushel)", một số theo "1/2 giạ", một số theo từng quả, một số theo cân nặng, và một số trong các thùng lớn với độ rộng khác nhau.

> Các quả bí ngô dường như rất khó cân đo một cách nhất quán

Khi tìm hiểu sâu hơn về dữ liệu gốc, thú vị là bất cứ thứ gì có "Unit of sale" là 'EACH' hoặc 'PER BIN' cũng có "package" dùng đơn vị inch, theo thùng, hoặc 'mỗi'. Các quả bí ngô dường như rất khó cân đo một cách nhất quán, vì vậy hãy lọc chúng bằng cách chỉ chọn các quả bí ngô có chuỗi 'bushel' trong cột "Package" của chúng.

1. Thêm một bộ lọc vào file, bên dưới đoạn code nhập file .csv:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Nếu bạn in dữ liệu ngay bây giờ, bạn có thể thấy rằng bạn chỉ nhận được khoảng 415 hàng dữ liệu chứa các quả bí ngô tính theo giạ.

### Từ từ đã! Còn một việc nữa

Bạn có nhận thấy rằng số lượng giạ thay đổi theo hàng không? Bạn cần chuẩn hóa để hiển thị giá trên mỗi giạ, vì vậy hãy thực hiện một số phép toán.

1. Thêm những dòng này vào sau đoạn code tạo new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Theo [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), trọng lượng của một giạ (bushel) phụ thuộc vào loại nông sản, vì đây là đơn vị đo thể tích. "Chẳng hạn, một giạ cà chua, được cho là nặng 56 pound (tương đương khoảng 25,4 kg)... Lá và rau xanh chiếm nhiều không gian hơn nhưng ít trọng lượng hơn, vì vậy một giạ rau chân vịt chỉ nặng 20 pound (~9,07kg)." Tất cả khá phức tạp! Chúng ta đừng bận tâm với việc chuyển đổi từ giạ sang khối lượng, mà thay vào đó hãy định giá theo giạ. Tuy nhiên, tất cả những nghiên cứu về giạ bí ngô này cho thấy việc hiểu bản chất dữ liệu quan trọng đến mức nào!

Bây giờ, bạn có thể phân tích giá bí ngô trên mỗi giạ. Nếu bạn in ra dữ liệu một lần nữa, bạn có thể thấy nó đã được chuẩn hóa.

## Trực quan hóa dữ liệu

Một phần vai trò của nhà khoa học dữ liệu là chứng minh chất lượng và bản chất của dữ liệu mà họ đang làm việc. Để làm điều này, họ thường tạo ra các hình ảnh trực quan, hoặc các biểu đồ, đồ thị, và bảng biểu, hiển thị các khía cạnh khác nhau của dữ liệu. Bằng cách này, họ có thể trực quan hóa các mối quan hệ mà nếu không trực quan hóa thì sẽ rất khó phát hiện.

[![ML for beginners - How to Visualize Data with Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML for beginners - How to Visualize Data with Matplotlib")

> 🎥 Click ảnh trên để xem video về cách trực quan hóa dữ liệu với Matplotlib

Việc trực quan hóa cũng có thể giúp xác định kỹ thuật học máy phù hợp nhất cho dữ liệu. Ví dụ, một biểu đồ phân tán có vẻ giống một đường thẳng cho thấy dữ liệu là một ứng viên tốt cho hồi quy tuyến tính.

Một thư viện trực quan hóa hoạt động tốt với Jupyter Notebook là [Matplotlib](https://matplotlib.org/) 

> Học thêm về trực quan hóa dữ liệu ở [khóa học của Microsoft](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Thực hành với Matplotlib

1. Khai báo thư viện Matplotlib ở đầu notebook:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Chạy lại (rerun) cả notebook.
1. Ở cuối notebook, thêm một ô để biểu thị dữ liệu dưới dạng hộp:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Một biểu đồ phân tán hiển thị mối quan hệ giữa giá và tháng](./images/scatterplot.png)

    Nó không đặc biệt hữu ích vì tất cả những gì nó làm là hiển thị trong dữ liệu của bạn dưới dạng điểm chênh lệch trong một tháng nhất định.

### Cải thiện

Để biểu đồ hiển thị dữ liệu hữu ích, bạn thường cần nhóm dữ liệu theo cách nào đó. Hãy thử tạo một biểu đồ trong đó trục y hiển thị các tháng và dữ liệu thể hiện sự phân bổ dữ liệu.

1. Thêm một ô để tạo biểu đồ cột được nhóm:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![A bar chart showing price to month relationship](./images/barchart.png)

    Đây là một cách trực quan hóa dữ liệu hữu ích hơn! Có vẻ như điều này cho thấy giá bí ngô cao nhất là vào tháng 9 và tháng 10. ---


