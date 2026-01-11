import json

def convert_fashioniq_to_json(input_txt, output_json):
    database = []
    
    try:
        with open(input_txt, 'r', encoding='utf-8') as f:
            for line in f:
                # Tách dòng dựa trên khoảng trắng hoặc tab
                parts = line.strip().split()
                
                # Kiểm tra nếu dòng có đủ ít nhất 2 thành phần (ASIN và URL)
                if len(parts) >= 2:
                    asin = parts[0]
                    url = parts[1]
                    
                    # Tạo dictionary chỉ với 2 field cần thiết
                    entry = {
                        "asin": asin,
                        "url": url
                    }
                    database.append(entry)
                    
        # Ghi ra file JSON
        with open(output_json, 'w', encoding='utf-8') as json_file:
            json.dump(database, json_file, indent=4, ensure_ascii=False)
            
        print(f"Thành công! Đã tạo file {output_json} với {len(database)} mục.")
        
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file đầu vào. Hãy kiểm tra lại đường dẫn.")

# Chạy thử với file của bạn
convert_fashioniq_to_json('toptee.txt', 'toptee.json')