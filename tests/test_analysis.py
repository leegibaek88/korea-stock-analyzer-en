import importlib.util
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import analyzer as en
import market_scanner as scanner
from listing_provider import normalize_listing


def prices(size=340):
    close = np.linspace(10000, 23000, size) + np.sin(np.arange(size) / 9) * 200
    return pd.DataFrame({'Open': close * .995, 'High': close * 1.01,
                         'Low': close * .985, 'Close': close,
                         'Volume': np.full(size, 100000)},
                        index=pd.bdate_range('2024-01-01', periods=size))


class EnglishEditionTests(unittest.TestCase):
    def test_english_and_native_names_resolve_same_code(self):
        listing = normalize_listing(pd.DataFrame([
            {'Code': '005930', 'Name': '\uc0bc\uc131\uc804\uc790', 'Market': 'KOSPI'},
            {'Code': '000660', 'Name': 'SK\ud558\uc774\ub2c9\uc2a4', 'Market': 'KOSPI'}]))
        for query in ['Samsung Electronics', 'samsung electronics', '\uc0bc\uc131\uc804\uc790', '005930']:
            self.assertEqual(en.resolve_stock_query(query, 'KOSPI', listing)['code'], '005930')

    def test_analysis_output_and_short_history(self):
        df = en.calculate_indicators(prices())
        with patch.object(en, 'load_stock_data', return_value=(prices(), prices())):
            result = en.analyze_stock('005930', name='Samsung Electronics')
        self.assertIn('Entry timing', result['text']['diagnosis'])
        self.assertFalse(re.search('[\uac00-\ud7a3]', str(result['text'])))
        short = en.calculate_minervini(df.tail(40).drop(columns=['MA20','MA50','MA150','MA200','VOL20','VOL50']).pipe(en.calculate_indicators))
        self.assertTrue(any(c['result'] == 'Insufficient data' for c in short['checks']))

    def test_daily_and_historical_mover_columns(self):
        snapshot = pd.DataFrame([{'Code':'005930','Name':'Samsung Electronics','Market':'KOSPI',
                                  'Close':20000,'ChagesRatio':2.1,'Volume':100000,'Amount':5e9,'Marcap':2e12}])
        with patch.object(scanner, 'market_snapshot', return_value=snapshot):
            daily = scanner.movers_top10_bundle_by_market('Daily')['KOSPI']['Rising']
        self.assertEqual(list(daily.columns), ['Rank','Stock','Analysis link','Last close','Return'])
        with patch.object(scanner.fdr,'DataReader',return_value=prices()):
            row=scanner._mover_candidate('005930','Samsung Electronics','KOSPI','2024-01-01',20)
        self.assertGreater(row['Return'],0)
        self.assertIn('analyze_code=005930',row['Analysis link'])

    def test_recommendation_and_case_render_schema(self):
        df=prices()
        with patch.object(scanner.fdr,'DataReader',return_value=df):
            row=scanner._analyze_candidate('005930','Samsung Electronics','KOSPI',en.calculate_indicators(df),'2024-01-01','Weekly')
            case=scanner._historical_case_candidate('005930','Samsung Electronics','KOSPI',en.calculate_indicators(df),'2024-01-01',60)
        self.assertIsNotNone(row)
        self.assertIn('Entry range', scanner._format_recommendation_frame(pd.DataFrame([row])).columns)
        if case:
            for key in ['Signal price','Signal score','Closing return','Lowest price return','Key signals','Analysis link']:
                self.assertIn(key,case)

    def test_numeric_parity_with_korean_source_when_available(self):
        path=Path(__file__).resolve().parents[2]/'stock_analyzer'/'analyzer.py'
        if not path.exists(): self.skipTest('Korean reference checkout not present')
        spec=importlib.util.spec_from_file_location('korean_reference',path)
        ko=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ko)
        for size in [90,220,340]:
            df=en.calculate_indicators(prices(size))
            benchmark=en.calculate_indicators(prices(size)*.9)
            results=[]
            for module in [ko,en]:
                m=module.calculate_minervini(df); o=module.calculate_oneil(df)
                w=module.calculate_weinstein(df); rs=module.calculate_relative_strength(df,benchmark)
                v=module.calculate_vcp(df)
                final=module.calculate_final_score(m,o,w,rs,v,df)
                results.append([m['score'],o['score'],w['score'],rs['score'],v['score'],final['total_score']])
            self.assertEqual(results[0],results[1])


if __name__=='__main__': unittest.main()
