#!/usr/bin/env python3
"""
🧪 SCRIPT DE VALIDAÇÃO - Verifica se as correções foram aplicadas corretamente

Uso:
    python validate_optimizations.py
"""

import os
import sys
import re

def check_file_for_pattern(filepath, pattern, description):
    """Verifica se um arquivo contém um padrão esperado."""
    if not os.path.exists(filepath):
        print(f"❌ {description}: Arquivo não encontrado ({filepath})")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if re.search(pattern, content):
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description}")
        return False

def main():
    print("\n" + "="*70)
    print("🔍 VALIDAÇÃO DAS OTIMIZAÇÕES DE MEMÓRIA RAM (SEM SAMPLING)")
    print("="*70 + "\n")
    
    checks = [
        # Analyze sweep results
        (
            "analyze_sweep_results.py",
            r"import gc",
            "1. Import gc em analyze_sweep_results.py"
        ),
        (
            "analyze_sweep_results.py",
            r"Pool\(processes=self\.n_workers,\s*maxtasksperchild=1\)",
            "2. Pool com maxtasksperchild=1 em analyze_all_parallel"
        ),
        (
            "analyze_sweep_results.py",
            r"use_parallel=False",
            "3. Paralelismo forçado sequencial em _analyze_lambda_worker"
        ),
        (
            "analyze_sweep_results.py",
            r"gc\.collect\(\)",
            "4. gc.collect() no finally de _analyze_lambda_worker"
        ),
        
        # Portfolio analyzer
        (
            "portfolio_utils/portfolio_analyzer.py",
            r"import gc",
            "5. Import gc em portfolio_analyzer.py"
        ),
        (
            "portfolio_utils/portfolio_analyzer.py",
            r"matplotlib\.use\('Agg'\)",
            "6. Backend Agg configurado para matplotlib"
        ),
        (
            "portfolio_utils/portfolio_analyzer.py",
            r"def _downcast_float_columns",
            "7. Função _downcast_float_columns implementada"
        ),
        (
            "portfolio_utils/portfolio_analyzer.py",
            r"_downcast_float_columns\(df\)",
            "8. Downcast chamado em _load_dataframe_flexible"
        ),
        (
            "portfolio_utils/portfolio_analyzer.py",
            r"# \u2b50 CR[ÍI]TICO: Modificar DataFrame \*in-place\*",
            "9. Comentário in-place em calculate_interpolation_errors (sem .copy())"
        ),
        (
            "portfolio_utils/portfolio_analyzer.py",
            r"plt\.hexbin\(df_logs\[\"risk\"\],\s*df_logs\[\"expected_return\"\]",
            "10. plt.hexbin implementado em _plot_efficient_vs_all (sem scatter)"
        ),
        (
            "portfolio_utils/portfolio_analyzer.py",
            r"plt\.close\('all'\)\s+gc\.collect\(\)",
            "11. plt.close('all') e gc.collect() ao final de _plot_efficient_vs_all"
        ),
        (
            "portfolio_utils/portfolio_analyzer.py",
            r"del df_logs, pareto_frontier, efficient_frontier\s+gc\.collect\(\)",
            "12. gc.collect() ao final de analyze_portfolio_results"
        ),
    ]
    
    passed = 0
    failed = 0
    
    for filepath, pattern, description in checks:
        full_path = os.path.join(os.path.dirname(__file__), filepath)
        if check_file_for_pattern(full_path, pattern, description):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTADO: {passed}/{len(checks)} verificações passaram")
    if failed == 0:
        print("✅ TODAS AS OTIMIZAÇÕES FORAM IMPLEMENTADAS CORRETAMENTE!")
    else:
        print(f"❌ {failed} verificações falharam - revisar código")
    print("="*70 + "\n")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
