---
layout: post
title:  "Bài 3 : Trách nhiệm con người trong việc xây dựng AI"
date:   2024-04-30 16:00:00 +0700
author: "Phạm Bình An"
categories: 
---
 
![Tóm tắt về việc xây dựng AI có trách nhiệm trong một trang giấy]({{site.url}}/assets/images/ml-fairness.png)
> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

 
## Giới thiệu

Trong khóa này, bạn sẽ bắt đầu khám phá cách máy học có thể và đang tác động đến cuộc sống hàng ngày của chúng ta như thế nào. Ngay cả hiện nay, các hệ thống và mô hình vẫn tham gia vào các nhiệm vụ ra quyết định hàng ngày, chẳng hạn như chẩn đoán chăm sóc sức khỏe, phê duyệt khoản vay hoặc phát hiện gian lận. Vì vậy, điều quan trọng là các mô hình này hoạt động tốt để cung cấp kết quả đáng tin cậy. Tuy nhiên, AI có thể không đạt được kỳ vọng hoặc gây ra hậu quả quả không mong muốn. Đó là lý do tại sao việc hiểu và giải thích hành vi của một mô hình AI là điều cần thiết.

Hãy tưởng tượng điều gì có thể xảy ra khi dữ liệu bạn đang sử dụng để xây dựng các mô hình này thiếu thông tin về một nhóm người nhất định, chẳng hạn như chủng tộc, giới tính, quan điểm chính trị, tôn giáo hoặc đại diện một cách không cân đối cho những nhóm người đó. Thế còn khi đầu ra của mô hình được hiểu là có lợi cho một số nhân khẩu học thì sao? Hậu quả của việc áp dụng là gì? Ngoài ra, điều gì sẽ xảy ra khi mô hình có kết quả bất lợi và có hại cho con người? Ai chịu trách nhiệm về hành vi của hệ thống AI? Đây là một số câu hỏi chúng ta sẽ khám phá trong khóa học này.

Trong bài học này, bạn sẽ:

- Nâng cao nhận thức của bạn về tầm quan trọng của sự công bằng trong học máy 
- Làm quen với việc thực hành khám phá các ngoại lệ và các tình huống bất thường để đảm bảo độ tin cậy và an toàn
- Đạt được sự hiểu biết về nhu cầu trao quyền cho mọi người bằng cách thiết kế các hệ thống hòa nhập
- Khám phá tầm quan trọng của việc bảo vệ quyền riêng tư và bảo mật
- Thấy được tầm quan trọng của việc áp dụng phương pháp hộp kính để giải thích hành vi của các mô hình AI
- Hãy lưu ý đến tầm quan trọng của trách nhiệm giải trình trong việc xây dựng niềm tin vào hệ thống AI

## Bắt đầu

Tìm hiểu về AI có trách nhiệm [link](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Cách tiếp cận của Microsoft đối với AI có trách nhiệm](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Cách tiếp cận của Microsoft đối với AI có trách nhiệm")

> 🎥 Click ảnh trên để xem video: Cách tiếp cận của Microsoft đối với AI có trách nhiệm

## Sự công bằng (fairness)

Hệ thống AI nên đối xử công bằng với mọi người và tránh ảnh hưởng đến những nhóm người tương tự theo những cách khác nhau. Ví dụ: khi hệ thống AI cung cấp hướng dẫn về điều trị y tế, đăng ký khoản vay hoặc việc làm, chúng phải đưa ra khuyến nghị giống nhau cho những người có triệu chứng, hoàn cảnh tài chính hoặc trình độ chuyên môn tương tự. Mỗi người trong chúng ta đều mang trong mình những thành kiến di truyền ảnh hưởng đến quyết định và hành động của chúng ta. Những thành kiến này có thể được thể hiện rõ trong dữ liệu mà chúng tôi sử dụng để đào tạo hệ thống AI. Điều này đôi khi có thể xảy ra ngoài ý muốn. Thường rất khó để biết khi nào bạn đang đưa ra thiên vị trong dữ liệu.

**“Sự bất công”** (unfairness) bao gồm các tác động tiêu cực hoặc “tác hại” đối với một nhóm người, chẳng hạn như những người được xác định theo chủng tộc, giới tính, tuổi tác hoặc tình trạng khuyết tật. Những tác hại chính liên quan đến sự công bằng có thể được phân loại là:

- **Phân bổ** (allocation), ví dụ: nếu một giới tính hoặc dân tộc được ưa chuộng hơn giới tính hoặc dân tộc khác.
- **Chất lượng dịch vụ** (quality of service). Nếu bạn huấn luyện dữ liệu cho một kịch bản cụ thể nhưng thực tế lại phức tạp hơn nhiều, điều đó sẽ dẫn đến dịch vụ hoạt động kém. Ví dụ, một hộp đựng xà phòng rửa tay dường như không thể cảm nhận được những người có làn da sẫm màu và do đó không chảy xà phòng cho người đó. [Tham khảo](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Chê bai** (denigration). Bình luận và dán nhãn một cách không công bằng cho một cái gì đó hoặc một ai đó. Ví dụ: một công nghệ dán nhãn hình ảnh đã dán nhãn sai cho hình ảnh của những người da đen là khỉ đột.
- **Đại diện quá mức hoặc quá ít** (over or under-representation). Ý tưởng cho rằng một nhóm người nhất định không được không làm một nghề nhất định.
- **Định kiến**. Liên kết một nhóm nhất định với các thuộc tính được gán trước. Ví dụ: hệ thống dịch ngôn ngữ giữa tiếng Anh và tiếng Thổ Nhĩ Kỳ có thể không chính xác do các từ có mối liên hệ khuôn mẫu với giới tính.

![dịch sang tiếng Thổ Nhĩ Kỳ]({{site.url}}/assets/images/gender-bias-translate-en-tr.png)
> dịch sang tiếng Thổ Nhĩ Kỳ

![dịch ngược lại tiếng Anh]({{site.url}}/assets/images/gender-bias-translate-tr-en.png)
> dịch ngược lại sang tiếng Anh

Khi thiết kế và thử nghiệm hệ thống AI, chúng ta cần đảm bảo rằng AI công bằng và không được lập trình để đưa ra những quyết định thiên vị hoặc phân biệt đối xử, điều mà con người cũng bị cấm đưa ra. Đảm bảo sự công bằng trong AI và học máy vẫn là một thách thức kỹ thuật xã hội phức tạp.

### Độ tin cậy và an toàn

Để tạo dựng niềm tin, hệ thống AI cần phải đáng tin cậy, an toàn và nhất quán trong các điều kiện bình thường và bất ngờ. Điều quan trọng là phải biết hệ thống AI sẽ hoạt động như thế nào trong nhiều tình huống khác nhau, đặc biệt là các trường hợp ngoại lệ. Khi xây dựng các giải pháp AI, cần phải tập trung đáng kể vào cách xử lý nhiều tình huống khác nhau. Ví dụ, ô tô tự lái cần đặt sự an toàn của con người lên hàng đầu. Do đó, AI vận hành ô tô cần xem xét tất cả các tình huống có thể xảy ra mà ô tô có thể gặp phải như ban đêm, giông bão hoặc bão tuyết, trẻ em chạy qua đường, vật nuôi, vật cản trên đường, v.v. 

> [🎥 Nhấn vào đây để xem video ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Tính toàn diện

Hệ thống AI nên được thiết kế để hỗ trợ cho mọi người. Khi thiết kế và triển khai hệ thống AI, các nhà khoa học dữ liệu và nhà phát triển AI sẽ xác định và giải quyết các rào cản tiềm ẩn trong hệ thống có thể vô tình loại trừ con người. Ví dụ, trên thế giới có 1 tỷ người khuyết tật. Với sự tiến bộ của AI, họ có thể tiếp cận nhiều loại thông tin và cơ hội dễ dàng hơn trong cuộc sống hàng ngày. Bằng cách giải quyết các rào cản, nó tạo ra cơ hội đổi mới và phát triển các sản phẩm AI với trải nghiệm tốt hơn mang lại lợi ích cho mọi người.

> [🎥 Xem video: inclusiveness in AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### An toàn và quyền riêng tư

Các hệ thống AI nên an toàn và tôn trọng quyền riêng tư. Mọi người ít tin tưởng các hệ thống khiến quyền riêng tư, thông tin, tính mạng của họ gặp rủi ro. Khi huấn luyện các mô hình học máy, chúng ta dựa vào dữ liệu để tạo ra kết quả tốt nhất. Khi đó, cần xem xét nguồn gốc và tính toàn vẹn của dữ liệu. Chẳng hạn, dữ liệu đó là công khai hay do người dùng gửi lên? Trong khi làm việc với dữ liệu, điều quan trọng là phải phát triển hệ thống AI có thể bảo vệ thông tin bí mật và chống lại các cuộc tấn công. Khi AI phổ biến hơn, việc bảo vệ quyền riêng tư và bảo mật thông tin cá nhân và doanh nghiệp ngày càng quan trọng và phức tạp hơn. Các vấn đề về quyền riêng tư và bảo mật dữ liệu đòi hỏi nhà phát triển AI phải đặc biệt chú ý vì quyền truy cập vào dữ liệu là điều cần thiết để hệ thống AI đưa ra những dự đoán và quyết định chính xác và sáng suốt về con người.

> [🎥 Click the here for a video: security in AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Với tư cách là một ngành, chúng ta đã đạt được những tiến bộ đáng kể trong Quyền riêng tư và Bảo mật, được thúc đẩy đáng kể bởi các quy định như GDPR (Quy định Bảo vệ Dữ liệu Chung của EU).
- Tuy nhiên, với các hệ thống AI, chúng ta phải công nhận sự căng thẳng giữa nhu cầu về nhiều dữ liệu cá nhân hơn để làm cho các hệ thống trở nên cá nhân hóa và hiệu quả hơn - và quyền riêng tư.
- Giống như với sự ra đời của máy tính kết nối với internet, chúng ta cũng đang chứng kiến sự gia tăng đáng kể số lượng vấn đề bảo mật liên quan đến AI.
- Đồng thời, chúng ta cũng thấy AI được sử dụng để cải thiện bảo mật. Ví dụ, hầu hết các trình quét virus hiện đại ngày nay đều được điều khiển bởi các AI heuristics.
- Chúng ta cần đảm bảo rằng các quy trình Khoa học Dữ liệu của chúng ta hài hòa với các thực hành về quyền riêng tư và bảo mật mới nhất. 


### Tính minh bạch
Hệ thống trí tuệ nhân tạo cần được hiểu rõ. Một yếu tố quan trọng của tính minh bạch là việc giải thích hành vi của chúng và các thành phần liên quan. Để nâng cao sự hiểu biết về hệ thống trí tuệ nhân tạo, các bên liên quan cần phải thấu hiểu cách chúng hoạt động và lý do tại sao chúng hoạt động như vậy. Điều này giúp phát hiện các vấn đề về hiệu suất, an toàn, và quyền riêng tư, cũng như tránh thiên hướng thiên vị và kết quả không mong muốn. Người sử dụng hệ thống trí tuệ nhân tạo cũng nên trung thực và cởi mở về việc triển khai chúng, bao gồm cả nhận biết các hạn chế của hệ thống mà họ sử dụng. Ví dụ, nếu một ngân hàng sử dụng trí tuệ nhân tạo để hỗ trợ quyết định về việc cho vay, quan trọng là phải đánh giá kết quả và hiểu rõ những dữ liệu nào ảnh hưởng đến các khuyến nghị của hệ thống. Với việc chính phủ bắt đầu quy định về trí tuệ nhân tạo trong các ngành công nghiệp, các chuyên gia dữ liệu và tổ chức cần phải giải thích xem liệu hệ thống trí tuệ nhân tạo có tuân thủ các quy định hay không, đặc biệt khi có những kết quả không mong muốn xuất hiện.

> [🎥 Video : Minh bạch trong AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Vì AI rất phức tạp, rất khó để hiểu cách chúng hoạt động và giải thích kết quả tạo ra
- Sự thiếu hiểu biết này ảnh hưởng đến cách quản lý, vận hành và viết tài liệu cho các hệ thống này.
- Quan trọng hơn, sự thiếu hiểu biết này ảnh hưởng đến các quyết định được đưa ra bằng cách sử dụng kết quả do AI tạo ra.

### Trách nhiệm 

Những người thiết kế và triển khai các hệ thống trí tuệ nhân tạo phải chịu trách nhiệm về cách hệ thống của họ hoạt động. Sự cần thiết của trách nhiệm này đặc biệt quan trọng đối với các công nghệ nhạy cảm như nhận dạng khuôn mặt. Gần đây, đã có một sự gia tăng nhu cầu về công nghệ nhận dạng khuôn mặt, đặc biệt từ các lực lượng thực thi pháp luật, những người nhìn thấy tiềm năng của công nghệ trong việc tìm kiếm trẻ em mất tích. Tuy nhiên, những công nghệ này có thể tiềm ẩn nguy cơ bị chính quyền sử dụng để hạn chế các quyền tự do của công dân, bằng cách cho phép giám sát liên tục của những cá nhân cụ thể. Do đó, các nhà khoa học dữ liệu và tổ chức cần phải chịu trách nhiệm về cách hệ thống trí tuệ nhân tạo của họ ảnh hưởng đến cá nhân hoặc xã hội.

[![Nhà nghiên cứu AI hàng đầu cảnh báo về giám sát hàng loạt qua nhận dạng khuôn mặt]({{site.url}}/assets/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft's Approach to Responsible AI")

> 🎥 Click ảnh trên để xem video : Nhà nghiên cứu AI hàng đầu cảnh báo về việc giám sát hàng loạt thông qua nhận dạng khuôn mặt

Cuối cùng, một trong những câu hỏi lớn nhất đối với thế hệ chúng ta, với tư cách là thế hệ đầu tiên đưa AI đến với xã hội, là làm thế nào để đảm bảo rằng máy tính sẽ có trách nhiệm với mọi người và làm thế nào để đảm bảo rằng những người thiết kế máy tính vẫn có trách nhiệm với mọi người khác.

##Đánh giá tác động

Trước khi đào tạo mô hình học máy, điều quan trọng là phải tiến hành đánh giá tác động để hiểu mục đích của hệ thống AI; mục đích sử dụng là gì; nơi nó sẽ được triển khai; và ai sẽ tương tác với hệ thống. Những điều này rất hữu ích cho người đánh giá hoặc người kiểm tra đánh giá hệ thống để biết những yếu tố nào cần xem xét khi xác định các rủi ro tiềm ẩn và hậu quả dự kiến.

Sau đây là các lĩnh vực trọng tâm khi tiến hành đánh giá tác động:

* **Tác động bất lợi đến cá nhân**. Việc nhận thức được mọi hạn chế hoặc yêu cầu, việc sử dụng không được hỗ trợ hoặc bất kỳ hạn chế đã biết nào cản trở hiệu suất của hệ thống là điều quan trọng để đảm bảo rằng hệ thống không được sử dụng theo cách có thể gây hại cho các cá nhân.
* **Yêu cầu về dữ liệu**. Việc hiểu rõ về cách thức và vị trí hệ thống sẽ sử dụng dữ liệu sẽ cho phép người đánh giá khám phá mọi yêu cầu về dữ liệu mà bạn cần lưu ý (ví dụ: các quy định về dữ liệu GDPR hoặc HIPPA). Ngoài ra, hãy kiểm tra xem nguồn hoặc số lượng dữ liệu có đáng kể cho việc đào tạo hay không.
* **Tóm tắt tác động**. Thu thập danh sách các tác hại tiềm ẩn có thể phát sinh từ việc sử dụng hệ thống. Trong suốt vòng đời ML, hãy xem xét xem các vấn đề được xác định có được giảm thiểu hoặc giải quyết hay không.
* **Mục tiêu áp dụng** cho từng nguyên tắc trong số sáu nguyên tắc cốt lõi. Đánh giá xem các mục tiêu của mỗi nguyên tắc có được đáp ứng hay không và có bất kỳ khoảng trống nào không.


## Gỡ lỗi (debug) AI

Tương tự như debug một ứng dụng phần mềm, debug hệ thống AI là một quá trình cần thiết để xác định và giải quyết các vấn đề trong hệ thống. Có nhiều yếu tố có thể ảnh hưởng đến việc một mô hình không hoạt động như mong đợi hoặc không có trách nhiệm. Hầu hết các số liệu hiệu suất mô hình truyền thống là tổng hợp định lượng về hiệu suất của mô hình, không đủ để phân tích xem mô hình vi phạm các nguyên tắc AI có trách nhiệm như thế nào. Hơn nữa, mô hình học máy là một hộp đen khiến cho việc hiểu điều gì thúc đẩy kết quả của nó hoặc đưa ra lời giải thích khi nó mắc lỗi trở nên khó khăn. Ở phần sau của khóa học này, chúng ta sẽ tìm hiểu cách sử dụng bảng thông tin AI có trách nhiệm để giúp gỡ lỗi hệ thống AI. Trang tổng quan cung cấp một công cụ toàn diện để các nhà khoa học dữ liệu và nhà phát triển AI thực hiện:

* **Phân tích lỗi**. Để xác định sự phân bố lỗi của mô hình có thể ảnh hưởng đến tính công bằng hoặc độ tin cậy của hệ thống.
* **Tổng quan về mô hình**. Để khám phá xem có sự khác biệt ở đâu về hiệu suất của mô hình giữa các nhóm dữ liệu.
* **Phân tích dữ liệu**. Để hiểu cách phân phối dữ liệu và xác định bất kỳ sai lệch tiềm ẩn nào trong dữ liệu có thể dẫn đến các vấn đề về tính công bằng, tính toàn diện và độ tin cậy.
* **Khả năng diễn giải mô hình**. Để hiểu những gì ảnh hưởng hoặc ảnh hưởng đến dự đoán của mô hình. Điều này giúp giải thích hành vi của mô hình, điều này rất quan trọng đối với tính minh bạch và trách nhiệm giải trình.


## 🚀 Thử thách
 
Để tránh gây ra tác hại ngay từ đầu, chúng ta nên:

- có sự đa dạng về nền tảng và quan điểm giữa những người làm việc trên hệ thống
- đầu tư vào các bộ dữ liệu (dataset) phản ánh sự đa dạng của xã hội chúng ta
- phát triển các phương pháp tốt hơn trong suốt vòng đời máy học để phát hiện và sửa chữa AI có trách nhiệm khi có vấn đề

Hãy nghĩ về các tình huống thực tế trong đó sự không đáng tin cậy của mô hình được thể hiện rõ trong quá trình xây dựng và sử dụng mô hình. Chúng ta nên xem xét điều gì khác?

## Ôn tập & Tự học
 
Trong bài học này, bạn đã học được một số khái niệm cơ bản về sự công bằng và bất công trong học máy.
 
Xem workshop này để tìm hiểu sâu hơn về các chủ đề:

- Theo đuổi AI có trách nhiệm: Áp dụng các nguyên tắc vào thực tiễn của Besmira Nushi, Mehrnoosh Sameki và Amit Sharma

[![Hộp công cụ AI có trách nhiệm: Khung nguồn mở để xây dựng AI có trách nhiệm](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "Hộp công cụ RAI: Khung nguồn mở để xây dựng AI có trách nhiệm")

> 🎥 Nhấp vào hình ảnh bên trên để xem video: Hộp công cụ RAI: Khung nguồn mở để xây dựng AI có trách nhiệm của Besmira Nushi, Mehrnoosh Sameki và Amit Sharma

Ngoài ra, hãy đọc:

- Trung tâm tài nguyên RAI của Microsoft: [Tài nguyên AI có trách nhiệm – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Nhóm nghiên cứu FATE của Microsoft: [FATE: Công bằng, Trách nhiệm giải trình, Minh bạch và Đạo đức trong AI - Nghiên cứu của Microsoft](https://www.microsoft.com/research/theme/fate/)

Hộp công cụ RAI:

- [Kho lưu trữ GitHub của Hộp công cụ AI có trách nhiệm](https://github.com/microsoft/responsible-ai-toolbox)

Đọc về các công cụ của Azure Machine Learning:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)