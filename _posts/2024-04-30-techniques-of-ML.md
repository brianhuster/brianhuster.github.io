---
layout: post
title:  "Bài 4 : Các kỹ thuật trong học máy"
date:   2024-04-30 16:00:00 +0700
author: "Phạm Bình An"
categories: 
---

Xem thêm các bài học khác của khóa **học máy cơ bản** tại [đây]({{site.url}}/2024/05/04/intro-to-regression.html)

Quá trình xây dựng, sử dụng và bảo trì các mô hình machine learning cũng như dữ liệu chúng sử dụng rất khác so với nhiều quy trình phát triển khác. Trong bài học này, chúng tôi sẽ làm sáng tỏ quy trình và phác thảo các kỹ thuật chính mà bạn cần biết. Bạn sẽ:

- Hiểu các quy trình làm nền tảng cho việc học máy ở mức độ cao.
- Khám phá các khái niệm cơ bản như 'mô hình' (model), 'dự đoán' (prediction) và 'dữ liệu huấn luyện' (training data).

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Nhấn ảnh trên để xem video của bài học này

## Giới thiệu

Ở cấp độ cao, tạo quá trình học máy (ML) bao gồm một số bước:

1. **Quyết định câu hỏi**. Vấn đề mà mô hình ML cần giải quyết
2. **Thu thập và chuẩn bị dữ liệu**. Để có thể trả lời câu hỏi của bạn, bạn cần dữ liệu. Chất lượng và đôi khi số lượng dữ liệu của bạn sẽ quyết định mức độ bạn có thể trả lời câu hỏi ban đầu của mình. Trực quan hóa dữ liệu là một khía cạnh quan trọng của giai đoạn này. Giai đoạn này cũng bao gồm việc chia dữ liệu thành tập huấn luyện (training) và thử nghiệm (testing) để xây dựng mô hình.
3. **Chọn phương pháp huấn luyện**. Tùy thuộc vào câu hỏi và tính chất của dữ liệu, bạn cần chọn cách bạn muốn huấn luyện mô hình để phản ánh tốt nhất dữ liệu của mình và đưa ra dự đoán chính xác dựa trên dữ liệu đó. Đây là một phần trong quy trình ML của bạn đòi hỏi chuyên môn cụ thể và thường cần lượngthí nghiệm đáng kể.
4. **Huấn luyện mô hình**. Khi sử dụng dữ liệu huấn luyện của mình, bạn sẽ sử dụng nhiều thuật toán khác nhau để huấn luyện mô hình nhằm nhận dạng các mẫu trong dữ liệu. Mô hình có thể tận dụng các trọng số bên trong có thể được điều chỉnh để ưu tiên một số phần dữ liệu nhất định so với các phần khác để xây dựng một mô hình tốt hơn.
5. **Đánh giá mô hình**. Bạn sử dụng dữ liệu chưa từng thấy trước đây (dữ liệu thử nghiệm của bạn) từ bộ đã thu thập để đánh giá hiệu quả mô hình.
6. **Điều chỉnh tham số**. Dựa trên hiệu suất của mô hình, bạn có thể lặp lại quy trình bằng cách sử dụng các tham số hoặc biến khác nhau để kiểm soát hành vi của các thuật toán được sử dụng để huấn luyện mô hình.
7. **Dự đoán**. Sử dụng đầu vào mới để kiểm tra độ chính xác của mô hình của bạn.

## Đặt vấn đề

Máy tính đặc biệt có khả năng phát hiện các mẫu ẩn trong dữ liệu. Điều này rất hữu ích cho các nhà nghiên cứu có câu hỏi về một miền nhất định mà không thể dễ dàng trả lời bằng cách tạo một công cụ quy tắc dựa trên điều kiện. Ví dụ: được giao một nhiệm vụ tính toán, một nhà khoa học dữ liệu có thể xây dựng các quy tắc thủ công xung quanh tỷ lệ tử vong của người hút thuốc so với người không hút thuốc.

Tuy nhiên, khi nhiều biến số khác được đưa vào phương trình, mô hình ML có thể tỏ ra hiệu quả hơn trong việc dự đoán tỷ lệ tử vong trong tương lai dựa trên lịch sử sức khỏe trong quá khứ. Một ví dụ thú vị hơn có thể là đưa ra dự đoán thời tiết cho tháng 4 ở một địa điểm nhất định dựa trên dữ liệu bao gồm vĩ độ, kinh độ, biến đổi khí hậu, vị trí gần biển, mô hình dòng phản lực, v.v.

✅ [Slide](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20 June%2024%20Haupt_0.pdf) về các mô hình thời tiết này cung cấp thông tin lịch sử về việc sử dụng ML trong phân tích thời tiết.

## Trước khi xây dựng mô hình

Trước khi bắt đầu xây dựng mô hình của mình, có một số nhiệm vụ bạn cần phải hoàn thành. 

### Dữ liệu

Để giải quyết vấn đề, bạn cần có một lượng lớn dữ liệu đúng loại. Có hai điều bạn cần làm vào thời điểm này:

- **Thu thập dữ liệu**. Hãy ghi nhớ bài học trước về tính công bằng trong phân tích dữ liệu, hãy thu thập dữ liệu của bạn một cách cẩn thận. Hãy lưu ý đến nguồn của dữ liệu này, bất kỳ thành kiến cố hữu nào mà nó có thể có và ghi lại nguồn gốc của nó.
- **Chuẩn bị dữ liệu**. Có một số bước trong quá trình chuẩn bị dữ liệu. Bạn có thể cần đối chiếu dữ liệu và chuẩn hóa dữ liệu nếu nó đến từ nhiều nguồn khác nhau. Bạn có thể cải thiện chất lượng và số lượng của dữ liệu thông qua nhiều phương pháp khác nhau, chẳng hạn như chuyển đổi chuỗi thành số (như chúng tôi thực hiện trong [chương Phân cụm](../../5-Clustering/1-Visualize/README.md)). Bạn cũng có thể tạo dữ liệu mới, dựa trên dữ liệu gốc (như chúng tôi thực hiện trong [chương Phân loại](../../4-Classification/1-Introduction/README.md)). Bạn có thể dọn dẹp và chỉnh sửa dữ liệu (như chúng tôi sẽ làm trước bài học [chương Web App](../../3-Web-App/README.md)). Cuối cùng, bạn cũng có thể cần phải chọn ngẫu nhiên và xáo trộn nó, tùy thuộc vào kỹ thuật huấn luyện của bạn.

✅ Sau khi thu thập và xử lý dữ liệu của bạn, hãy dành chút thời gian để xem liệu hình dạng của dữ liệu có cho phép bạn giải quyết vấn đề không. Có thể dữ liệu sẽ không hoạt động tốt trong vấn đề của bạn, như chúng ta sẽ khám phá trong chương [Phân cụm](../../5-Clustering/1-Visualize/README.md)!

### Đặc trưng và mục tiêu

[Đặc trưng](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) là thuộc tính có thể đo lường được trong dữ liệu của bạn. Trong nhiều bộ dữ liệu, nó được biểu thị dưới dạng tiêu đề cột như 'ngày' 'kích thước' hoặc 'màu sắc'. Biến đặc trưng của bạn, thường được biểu thị dưới dạng `X` trong mã, biểu thị biến đầu vào sẽ được sử dụng để huấn luyện mô hình.

Mục tiêu là thứ bạn đang cố gắng dự đoán. Mục tiêu thường được biểu thị bằng `y` trong mã, thể hiện câu trả lời cho câu hỏi bạn đang cố gắng hỏi về dữ liệu của mình, chẳng hạn: vào tháng 12, quả bí ngô **màu** nào sẽ rẻ nhất? ở San Francisco, khu vực nào sẽ có giá bất động sản tốt nhất **? Đôi khi mục tiêu còn được gọi là thuộc tính nhãn.

### Chọn biến đặc trưng của bạn

🎓 **Lựa chọn đặc điểm và trích xuất đặc điểm** Làm sao bạn biết nên chọn biến nào khi xây dựng mô hình? Có thể bạn sẽ trải qua quá trình chọn hay trích xuất đặc trưng để chọn các biến phù hợp cho mô hình hoạt động hiệu quả nhất. Tuy nhiên, chúng không giống nhau: "Trích xuất đặc trưng tạo ra các đặc trưng mới từ các chức năng của các đặc trưng ban đầu, trong khi lựa chọn tính đặc trưng trả về một tập hợp con các tính năng." ([nguồn](https://wikipedia.org/wiki/Feature_selection))

### Trực quan hóa dữ liệu của bạn

Một khía cạnh quan trọng trong bộ công cụ của nhà khoa học dữ liệu là khả năng trực quan hóa dữ liệu bằng cách sử dụng một số thư viện như Seaborn hoặc MatPlotLib. Việc trình bày dữ liệu của bạn một cách trực quan có thể cho phép bạn khám phá các mối tương quan ẩn mà bạn có thể tận dụng. Hình ảnh trực quan của bạn cũng có thể giúp bạn phát hiện ra dữ liệu sai lệch hoặc không cân bằng (như chúng tôi khám phá trong [Phân lớp](../../4-Classification/2-Classifiers-1/README.md)).

### Chia tập dữ liệu của bạn

Trước khi đào tạo, bạn cần chia tập dữ liệu của mình thành hai hoặc nhiều phần có kích thước lệch nhau mà vẫn thể hiện tốt dữ liệu.

- **Tập huấn luyện** dùng để huấn luyện mô hình và thường chiếm phần lớn bộ dữ liệu (thường là 80%)
- **Tập thử nghiệm** dùng để thử nghiệm mô hình đã học
- **Tập tối ưu** dùng để tối ưu hóa tham số (nếu có)

## Xây dựng mô hình

### Xác định phương pháp huấn luyện

Bạn sẽ chọn phương pháp huấn luyện dựa vào vấn đề cần giải quyết và đặc điểm của bộ dữ liệu. [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) cung cấp nhiều cách huấn luyện một mô hình. Tùy vào kinh nghiệm của bạn, bạn có thể phải thử nhiều cách khác nhau để có thể xây dựng được mô hinh tốt nhất. 

### Huấn luyện mô hình

Với dữ liệu có sẵn, bạn đã sẵn sàng 'khớp' (fit) nó để tạo mô hình. Bạn sẽ nhận thấy rằng trong nhiều thư viện học máy, bạn sẽ tìm thấy mã 'model.fit' - đó là lúc bạn gửi biến đặc trưng của dữ liệu dưới dạng một mảng các giá trị (thường là 'X') và một biến mục tiêu (thường là 'y').

### Đánh giá mô hình

Sau khi quá trình đào tạo hoàn tất (có thể mất nhiều lần vòng lặp hoặc 'epoch' để đào tạo một mô hình lớn), bạn có thể đánh giá chất lượng của mô hình bằng cách sử dụng tập dữ liệu thử nghiệp. Tập này là tập con của bộ dữ liệu gốc mà mô hình sử dụng trước đó. Bạn có thể in ra bảng số liệu về chất lượng mô hình của mình.

🎓 **Độ khớp mô hình (Model fitting)**

Trong bối cảnh học máy, độ khớp mô hình đề cập đến độ chính xác của chức năng cơ bản của mô hình khi nó cố gắng phân tích dữ liệu mà nó không quen thuộc.

🎓 **Chưa khớp (underfitting)** và **quá khớp (overfitting)** là những vấn đề phổ biến làm giảm chất lượng của mô hình.

Quá khớp xảy ra khi mô hình học quá nhiều từ dữ liệu đào tạo, bao gồm cả nhiễu và chi tiết không cần thiết. Kết quả là mô hình sẽ hoạt động rất tốt trên dữ liệu đào tạo nhưng không thể tổng quát hóa tốt trên dữ liệu chưa từng gặp (dữ liệu kiểm thử). Điều này giống như một sinh viên học thuộc lòng tất cả các câu hỏi trong đề thi mẫu nhưng không thể giải quyết được khi gặp câu hỏi mới trong đề thi thật.

Chưa khớp xảy ra khi mô hình chưa học đủ từ dữ liệu đào tạo, không thể nắm bắt được mối quan hệ giữa các đặc trưng và mục tiêu. Kết quả là mô hình sẽ hoạt động kém cả trên dữ liệu đào tạo và dữ liệu kiểm thử. Điều này giống như một sinh viên không học đủ kiến thức để giải quyết các câu hỏi trong đề thi. 

![mô hình quá khớp]({{site.url}}/assets/images/classic-ML-course/overfitting.png)
> Hình ảnh của [Jen Looper](https://twitter.com/jenlooper)

## Điều chỉnh tham số

Sau khi quá trình huấn luyện ban đầu của bạn hoàn tất, hãy quan sát chất lượng của mô hình và xem xét cải thiện bằng cách điều chỉnh 'siêu tham số' (hyperparameter) của nó. Đọc thêm về [trong tài liệu](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Dự đoán

Đây là thời điểm bạn có thể dùng dữ liệu hoàn toàn mới để kiểm tra độ chính xác của mô hình, hoặc đưa mô hình vào ứng dụng thực tế. 

Xem thêm các bài học khác của khóa **học máy cơ bản** tại [đây]({{site.url}}/2024/05/04/intro-to-regression.html)

