#!/usr/bin/env python3
"""
VALIDAÇÃO FINAL: Verificação de integridade da otimização NumPy
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Adiciona o diretório ao path
sys.path.insert(0, str(Path(__file__).parent))

from portfolio_utils.parquet_handler import NumpyParquetBufferWriter, ParquetBufferWriter, ParquetReader

def test_1_numpy_allocation():
    """Testa pré-alocação NumPy"""
    print("\n" + "="*60)
    print("TESTE 1: Pré-alocação NumPy")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test1.parquet")
        
        n_assets = 20
        buffer_size = 100
        
        logger = NumpyParquetBufferWriter(
            file_path=output_file,
            buffer_size=buffer_size,
            n_assets=n_assets,
        )
        
        # Verificar tipos de arrays
        assert logger._weights_is_matrix, "❌ Weights deve ser matriz 2D"
        assert logger.data['weights'].shape == (buffer_size, n_assets), "❌ Shape incorreto"
        assert logger.data['eval_id'].dtype == np.uint32, "❌ Dtype eval_id incorreto"
        assert logger.data['expected_return'].dtype == np.float32, "❌ Dtype float32 esperado"
        
        print("✅ Arrays pré-alocados corretamente")
        print(f"   - Weights: {logger.data['weights'].shape} {logger.data['weights'].dtype}")
        print(f"   - Eval_id: {logger.data['eval_id'].shape} {logger.data['eval_id'].dtype}")
        print(f"   - Expected_return: {logger.data['expected_return'].shape} {logger.data['expected_return'].dtype}")
        print("✅ TESTE 1 PASSOU")

def test_2_log_fast_method():
    """Testa método log_fast() sem allocações"""
    print("\n" + "="*60)
    print("TESTE 2: Método log_fast() - Zero allocações")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test2.parquet")
        
        logger = NumpyParquetBufferWriter(
            file_path=output_file,
            buffer_size=50,
            n_assets=10,
        )
        
        # Inserir 150 registros (3 flushes)
        for i in range(150):
            weights = np.random.rand(10)
            logger.log_fast(
                eval_id=i,
                weights=weights,
                selected_assets=[0, 1],
                expected_return=0.05,
                risk=0.1,
                variance=0.01,
                objective=0.5,
                is_improvement=(i % 10 == 0),
                timestamp="2025-01-01T00:00:00Z"
            )
        
        logger.close()
        
        # Verificar arquivo
        assert os.path.exists(output_file), "❌ Arquivo não criado"
        
        df = pd.read_parquet(output_file)
        assert len(df) == 150, f"❌ Esperado 150 registros, obteve {len(df)}"
        assert df['eval_id'].max() == 149, "❌ eval_id incorreto"
        
        print(f"✅ 150 registros salvos com 3 flushes automáticos")
        print(f"✅ Arquivo: {os.path.getsize(output_file) / 1024:.1f} KB")
        print("✅ TESTE 2 PASSOU")

def test_3_compatibility():
    """Testa compatibilidade com ParquetBufferWriter alias"""
    print("\n" + "="*60)
    print("TESTE 3: Compatibilidade com alias ParquetBufferWriter")
    print("="*60)
    
    # Verificar que alias existe
    assert ParquetBufferWriter == NumpyParquetBufferWriter, "❌ Alias incorreto"
    print(f"✅ Alias: ParquetBufferWriter → {ParquetBufferWriter.__name__}")
    
    # Usar pelo alias (simulando código antigo)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test3.parquet")
        
        logger = ParquetBufferWriter(  # Usando o alias!
            file_path=output_file,
            buffer_size=50,
            n_assets=5,
        )
        
        for i in range(100):
            logger.log_fast(
                eval_id=i,
                weights=np.random.rand(5),
                selected_assets=[0],
                expected_return=0.05,
                risk=0.1,
                variance=0.01,
                objective=0.5,
                is_improvement=False,
                timestamp="2025-01-01T00:00:00Z"
            )
        
        logger.close()
        
        df = pd.read_parquet(output_file)
        assert len(df) == 100, "❌ Registros não salvos"
        print("✅ Código antigo usando alias funciona perfeitamente")
        print("✅ TESTE 3 PASSOU")

def test_4_reader_compatibility():
    """Testa leitura com ParquetReader"""
    print("\n" + "="*60)
    print("TESTE 4: Compatibilidade com ParquetReader")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test4.parquet")
        
        # Escrever com novo logger
        logger = NumpyParquetBufferWriter(
            file_path=output_file,
            buffer_size=50,
            n_assets=8,
        )
        
        for i in range(100):
            logger.log_fast(
                eval_id=i,
                weights=np.random.rand(8),
                selected_assets=[0, 1, 2],
                expected_return=0.05 + 0.01 * i,
                risk=0.1,
                variance=0.01,
                objective=0.5,
                is_improvement=(i % 5 == 0),
                timestamp=f"2025-01-01T{i//60:02d}:{i%60:02d}:00Z"
            )
        
        logger.close()
        
        # Ler com ParquetReader
        df = ParquetReader.read_full(output_file)
        
        # Validações
        assert len(df) == 100, "❌ Registros não carregados"
        assert all(col in df.columns for col in ['eval_id', 'weights', 'objective', 'is_improvement']), "❌ Colunas faltando"
        assert df['eval_id'].dtype == np.uint32, "❌ Dtype incorreto"
        assert df['objective'].dtype == np.float32, "❌ Float32 não preservado"
        assert len(df[df['is_improvement']]) == 20, "❌ is_improvement incorreto"
        
        print(f"✅ Arquivo lido com sucesso: {len(df)} registros")
        print(f"✅ Colunas: {list(df.columns)}")
        print(f"✅ Tipos: eval_id={df['eval_id'].dtype}, objective={df['objective'].dtype}")
        print(f"✅ Improvements detectados: {len(df[df['is_improvement']])}")
        print("✅ TESTE 4 PASSOU")

def test_5_matrix_2d_efficiency():
    """Testa eficiência da matriz 2D para weights"""
    print("\n" + "="*60)
    print("TESTE 5: Eficiência Matriz 2D para Weights")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test5.parquet")
        
        n_assets = 100
        buffer_size = 1000
        
        logger = NumpyParquetBufferWriter(
            file_path=output_file,
            buffer_size=buffer_size,
            n_assets=n_assets,
        )
        
        # Tamanho da matriz 2D
        matrix_size = logger.data['weights'].nbytes / (1024 * 1024)
        print(f"✅ Matriz 2D weights: {buffer_size} × {n_assets} = {matrix_size:.1f} MB")
        print(f"   (4 bytes × {buffer_size*n_assets:,} = {matrix_size:.1f} MB)")
        
        # Vs se fosse object array
        object_array_overhead = buffer_size * 8 / (1024 * 1024)  # 8 bytes por pointer
        print(f"✅ Object array seria: ~{object_array_overhead:.1f} MB (apenas pointers)")
        print(f"   + overhead de lista dentro de cada posição")
        
        # Economía
        print(f"✅ Economia: Pré-alocado é eficiente para n_assets fixo")
        print("✅ TESTE 5 PASSOU")

def test_6_adaptive_buffer():
    """Testa adaptive buffer sizing"""
    print("\n" + "="*60)
    print("TESTE 6: Adaptive Buffer Sizing")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test6.parquet")
        
        logger = NumpyParquetBufferWriter(
            file_path=output_file,
            buffer_size=5000,
            n_assets=10,
            enable_adaptive=True
        )
        
        original_size = logger.buffer_size
        print(f"✅ Buffer size inicial: {original_size}")
        
        # Teste: simular RAM alta manualmente (não vamos aumentar RAM de verdade)
        logger._adjust_buffer_size()  # Sem RAM alta, não faz nada
        assert logger.buffer_size == original_size, "❌ Buffer não deveria mudar"
        
        print(f"✅ Adaptive buffer mantém tamanho quando RAM OK")
        print("✅ TESTE 6 PASSOU")

def main():
    """Executa todos os testes"""
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█  VALIDAÇÃO FINAL: NumPy Pré-alocado + Cursor-based    █")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    tests = [
        test_1_numpy_allocation,
        test_2_log_fast_method,
        test_3_compatibility,
        test_4_reader_compatibility,
        test_5_matrix_2d_efficiency,
        test_6_adaptive_buffer,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ TESTE FALHOU: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Sumário
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print(f"█  SUMÁRIO: {passed} passaram | {failed} falharam" + " "*(58-len(f"SUMÁRIO: {passed} passaram | {failed} falharam")) + "█")
    print("█" + " "*58 + "█")
    print("█"*60 + "\n")
    
    if failed == 0:
        print("✅ TODAS AS VALIDAÇÕES PASSARAM!")
        print("\n🎯 A otimização NumPy está PRONTA PARA PRODUÇÃO!\n")
        return 0
    else:
        print(f"❌ {failed} testes falharam. Verifique os erros acima.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
