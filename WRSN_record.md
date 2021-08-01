**29/07/2021 - 9AM**: 
1. Thay đổi :
   - hàm reward, nếu hành động đưa ra kết quả là sạc 0s, reward = 0
     + Nếu phạt nặng quá (-5, -10) sẽ làm cho giá trị TD giao động lớn
   - giảm learning rate start xuống vì có hiện tượng hội tụ sớm
     + model trả về PLoss và ELoss là các giá trị nan vì hội tụ sớm (log(0) = nan)
   - sửa truncated_mu trong khoảng từ (0.1,5) thay vì (0,5)
2. Nhận xét:
   - Vẫn có hiện tượng các MC đến cùng một điểm sạc
   - Overlap khá nhiều

**29/07/2021 - 2PM**:
1. Thay đổi:
   - Trong worker, đặt điểm dừng cho việc lấy vector heuristic
   - Đưa hệ số vào H_policy trong hàm *H_get_heuristic_policy*, hiện tại là (0.35, 0.35, 0.4)
2. Nhận xét:
   - Chưa giải quyết được hiện tượng overlap

**30/07/2021 - 9AM**:
1. Thay đổi:
   - Thêm GLIP_GRAD để ổn định cập nhật
   - Đưa entropy loss vào cùng value loss
   - Chuyển Gradient Descent thành Ascent
   - Cho phép đạo hàm temporal difference với policy loss
   - Chuyển code thành multithreading trong hàm all_asynchronize của Worker_method
2. Nhận xét:
   - mu là tỉ lệ giữa policy_prob và behavior_prob, mu nên tiến từ 0 đến 1 để thể hiện rằng AC học theo Heuristic
   - Tuy nhiên mu đang có xu hướng giảm, có thể là lỗi code => cần debug

**30/07/2021 - 21PM**:
1. Thay đổi:
   - Tìm được lỗi logic trong hàm accumulate của worker
   - Thêm folder Model_weights/A3C để lưu lại các weights:
     + body_net's weights
     + critic_net's weights
     + actor_net's weights
   -> Các net này được train liên tục mỗi khi chạy mới
2. Nhiệm vụ:
   - Chuyển global_Optimizer ra ngoài vòng lặp for của nb_run