#!/bin/bash

# Automatic Video Processing Script
# 自動處理所有影片 - 無需任何參數
# Usage: ./auto_process_all.sh

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
INPUT_DIR="input"
OUTPUT_DIR="output/final"
TARGET_DIR="target_images"
VENV_PATH="venv/bin/python"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored messages
print_header() {
    echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║${NC}  $1"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_progress() {
    echo -e "${CYAN}[→]${NC} $1"
}

# Start
clear
print_header "自動影片處理系統 - Auto Video Processor"
echo ""

# Check if venv exists
if [ ! -f "$VENV_PATH" ]; then
    print_error "找不到虛擬環境: $VENV_PATH"
    exit 1
fi

# Get all video files
VIDEO_FILES=($(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" \) 2>/dev/null))

if [ ${#VIDEO_FILES[@]} -eq 0 ]; then
    print_error "在 $INPUT_DIR 目錄中找不到任何影片檔案"
    echo ""
    print_info "請將影片放到: $SCRIPT_DIR/$INPUT_DIR/"
    print_info "支援格式: .mp4, .avi, .mov"
    exit 1
fi

# Get all target face directories
TARGET_FACES=($(find "$TARGET_DIR" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | xargs -n 1 basename | sort))

if [ ${#TARGET_FACES[@]} -eq 0 ]; then
    print_error "在 $TARGET_DIR 目錄中找不到任何人臉資料夾"
    exit 1
fi

# Display summary
print_info "掃描結果:"
echo "  📁 影片數量: ${#VIDEO_FILES[@]}"
echo "  👤 人臉數量: ${#TARGET_FACES[@]}"
echo "  📊 將產生: $((${#VIDEO_FILES[@]} * ${#TARGET_FACES[@]})) 個影片"
echo ""

print_info "找到的影片:"
for VIDEO_FILE in "${VIDEO_FILES[@]}"; do
    VIDEO_NAME=$(basename "$VIDEO_FILE")
    VIDEO_SIZE=$(du -h "$VIDEO_FILE" | cut -f1)
    echo "  • $VIDEO_NAME ($VIDEO_SIZE)"
done
echo ""

print_info "找到的人臉:"
for TARGET_FACE in "${TARGET_FACES[@]}"; do
    FACE_COUNT=$(find "$TARGET_DIR/$TARGET_FACE" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)
    echo "  • $TARGET_FACE ($FACE_COUNT 張照片)"
done
echo ""

# Confirmation
print_warning "準備開始處理，按 Enter 繼續，或 Ctrl+C 取消..."
read -r

echo ""
print_header "開始處理 - Processing Started"
echo ""

# Counters
TOTAL_TASKS=$((${#VIDEO_FILES[@]} * ${#TARGET_FACES[@]}))
CURRENT_TASK=0
SUCCESS_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0

START_TIME=$(date +%s)

# Process each video with each target face
for VIDEO_FILE in "${VIDEO_FILES[@]}"; do
    VIDEO_PATH="$VIDEO_FILE"
    VIDEO_NAME=$(basename "$VIDEO_FILE")
    VIDEO_BASENAME="${VIDEO_NAME%.*}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_progress "處理影片: $VIDEO_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    for TARGET_FACE in "${TARGET_FACES[@]}"; do
        CURRENT_TASK=$((CURRENT_TASK + 1))
        OUTPUT_FILE="${VIDEO_BASENAME}_${TARGET_FACE}.mp4"
        OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILE"

        # Progress indicator
        PROGRESS_PERCENT=$((CURRENT_TASK * 100 / TOTAL_TASKS))
        print_info "[$CURRENT_TASK/$TOTAL_TASKS - ${PROGRESS_PERCENT}%] 套用人臉: $TARGET_FACE"

        # Check if output already exists
        if [ -f "$OUTPUT_PATH" ]; then
            print_warning "輸出檔案已存在，跳過: $OUTPUT_FILE"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            echo ""
            continue
        fi

        # Run face swap
        "$VENV_PATH" main.py \
            --source "$VIDEO_PATH" \
            --target "$TARGET_DIR/$TARGET_FACE" \
            --output "$OUTPUT_PATH" \
            --auto 2>&1 | grep -E "INFO|SUCCESS|ERROR|✓" | tail -20

        # Check if processing was successful
        if [ $? -eq 0 ] && [ -f "$OUTPUT_PATH" ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            FILE_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
            print_success "完成: $OUTPUT_FILE ($FILE_SIZE)"
        else
            FAILED_COUNT=$((FAILED_COUNT + 1))
            print_error "失敗: $TARGET_FACE"
        fi

        echo ""
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Final Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_header "處理完成 - All Processing Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

print_info "統計摘要:"
echo "  📥 處理影片數: ${#VIDEO_FILES[@]}"
echo "  👤 使用人臉數: ${#TARGET_FACES[@]}"
echo "  📊 總任務數: $TOTAL_TASKS"
echo ""
echo "  ✓ 成功: $SUCCESS_COUNT"
echo "  ✗ 失敗: $FAILED_COUNT"
echo "  ⊝ 跳過: $SKIPPED_COUNT"
echo ""
printf "  ⏱  總耗時: "
if [ $HOURS -gt 0 ]; then
    printf "%d 小時 " $HOURS
fi
if [ $MINUTES -gt 0 ] || [ $HOURS -gt 0 ]; then
    printf "%d 分鐘 " $MINUTES
fi
printf "%d 秒\n" $SECONDS
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    print_success "輸出影片已儲存在: $OUTPUT_DIR/"
    echo ""
    print_info "產生的影片列表:"
    ls -lh "$OUTPUT_DIR" | grep -E "${VIDEO_BASENAME}_" | awk '{printf "  • %s (%s)\n", $9, $5}'
fi

echo ""
if [ $FAILED_COUNT -eq 0 ]; then
    print_success "🎉 所有任務都已成功完成！"
else
    print_warning "⚠️  有 $FAILED_COUNT 個任務失敗，請檢查錯誤訊息"
fi

echo ""
exit 0
