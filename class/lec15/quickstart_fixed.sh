#!/bin/bash
# quickstart_fixed.sh - GRPO修复后的快速启动脚本

echo "=========================================="
echo "🎉 GRPO Loss修复版 - 快速启动"
echo "=========================================="
echo ""

# 切换到正确的目录
cd /apdcephfs/pig_data/MiniLLM

echo "📍 当前目录: $(pwd)"
echo ""

# 显示菜单
echo "请选择操作："
echo "  1) 运行修复验证测试"
echo "  2) 运行ORM模式训练（1 epoch）"
echo "  3) 运行PRM模式训练（1 epoch）"
echo "  4) 运行完整单元测试"
echo "  5) 查看修复文档"
echo "  6) 退出"
echo ""

read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "🧪 运行修复验证测试"
        echo "=========================================="
        python class/lec15/test_loss_fix.py
        ;;
    
    2)
        echo ""
        echo "=========================================="
        echo "🚀 运行ORM模式训练"
        echo "=========================================="
        echo "参数: --make_dummy --reward_mode orm --epochs 1"
        echo ""
        python class/lec15/grpo_demo.py \
            --make_dummy \
            --reward_mode orm \
            --epochs 1 \
            --batch_size 2 \
            --group_size 4 \
            --lr 1e-5
        ;;
    
    3)
        echo ""
        echo "=========================================="
        echo "🚀 运行PRM模式训练"
        echo "=========================================="
        echo "参数: --make_dummy --reward_mode prm --epochs 1"
        echo ""
        python class/lec15/grpo_demo.py \
            --make_dummy \
            --reward_mode prm \
            --epochs 1 \
            --batch_size 2 \
            --group_size 4 \
            --lr 1e-5
        ;;
    
    4)
        echo ""
        echo "=========================================="
        echo "🧪 运行完整单元测试"
        echo "=========================================="
        python class/lec15/test_grpo.py
        ;;
    
    5)
        echo ""
        echo "=========================================="
        echo "📚 修复文档"
        echo "=========================================="
        echo ""
        echo "可用文档："
        echo "  - FIX_SUMMARY.md          : 快速了解修复内容"
        echo "  - COMPLETE_FIX_SUMMARY.md : 完整修复说明"
        echo "  - LOSS_FIX.md             : 详细技术分析"
        echo "  - README.md               : 使用指南"
        echo ""
        echo "查看文档："
        echo "  cat class/lec15/FIX_SUMMARY.md"
        echo "  cat class/lec15/COMPLETE_FIX_SUMMARY.md"
        echo "  cat class/lec15/LOSS_FIX.md"
        echo ""
        ;;
    
    6)
        echo "退出"
        exit 0
        ;;
    
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ 完成"
echo "=========================================="
echo ""
echo "💡 提示："
echo "  - Loss应该在 0.5-5.0 范围内"
echo "  - KL散度应该 < 1.0"
echo "  - 优势值应该在 [-3, 3] 范围内"
echo ""
echo "📚 查看详细文档："
echo "  cat class/lec15/FIX_SUMMARY.md"
echo ""
