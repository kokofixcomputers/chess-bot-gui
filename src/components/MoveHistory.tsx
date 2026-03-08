import React, { useEffect, useRef } from 'react';
import type { HistoryEntry } from '../hooks/useChessGame';

interface Props {
  history: HistoryEntry[];
}

const PIECE_SYMBOL: Record<string, string> = {
  K: '♔', Q: '♕', R: '♖', B: '♗', N: '♘', P: '',
  k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '',
};

function formatMove(entry: HistoryEntry, index: number): string {
  const { alg, move } = entry;
  const pt = move.piece[1];
  const sym = pt === 'P' ? '' : PIECE_SYMBOL[pt] + ' ';
  const cap = move.captured ? '×' : '-';
  const from = alg.slice(0, 2);
  const to = alg.slice(2, 4);
  const prom = move.promotion ? `=${move.promotion}` : '';
  const castle = move.castling === 'K' ? 'O-O' : move.castling === 'Q' ? 'O-O-O' : null;
  if (castle) return castle;
  return `${sym}${from}${cap}${to}${prom}`;
}

export const MoveHistory: React.FC<Props> = ({ history }) => {
  const endRef = useRef<HTMLDivElement>(null);

  //useEffect(() => {
  //  endRef.current?.scrollIntoView({ behavior: 'smooth' });
  //}, [history.length]);

  const pairs: Array<[HistoryEntry, HistoryEntry | null]> = [];
  for (let i = 0; i < history.length; i += 2) {
    pairs.push([history[i], history[i + 1] ?? null]);
  }

  return (
    <div style={{
      background: 'rgba(255,255,255,0.025)',
      border: '1px solid rgba(255,255,255,0.07)',
      borderRadius: 14,
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
      flex: 1,
      minHeight: 0,
    }}>
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        fontSize: 11, fontWeight: 700, letterSpacing: '0.1em',
        textTransform: 'uppercase', color: 'rgba(255,255,255,0.3)',
      }}>
        Move History
      </div>

      <div style={{ overflowY: 'auto', padding: '8px 8px', flex: 1 }}>
        {pairs.length === 0 && (
          <div style={{ color: 'rgba(255,255,255,0.2)', fontSize: 13, padding: '8px 8px' }}>No moves yet</div>
        )}
        {pairs.map(([white, black], i) => (
          <div key={i} style={{
            display: 'grid',
            gridTemplateColumns: '28px 1fr 1fr',
            gap: 4,
            padding: '3px 4px',
            borderRadius: 6,
            background: i === pairs.length - 1 ? 'rgba(99,102,241,0.1)' : 'transparent',
          }}>
            <span style={{ color: 'rgba(255,255,255,0.2)', fontSize: 12, fontFamily: 'DM Mono, monospace', display: 'flex', alignItems: 'center' }}>
              {i + 1}.
            </span>
            <span style={{
              fontSize: 13, fontFamily: 'DM Mono, monospace',
              color: '#e2e8f0', padding: '2px 6px', borderRadius: 4,
              background: 'rgba(255,255,255,0.04)',
            }}>
              {formatMove(white, i * 2)}
            </span>
            {black && (
              <span style={{
                fontSize: 13, fontFamily: 'DM Mono, monospace',
                color: '#a78bfa', padding: '2px 6px', borderRadius: 4,
                background: 'rgba(167,139,250,0.08)',
              }}>
                {formatMove(black, i * 2 + 1)}
                {black.aiDepth && (
                  <span style={{ fontSize: 10, color: 'rgba(167,139,250,0.4)', marginLeft: 4 }}>
                    d{black.aiDepth}
                  </span>
                )}
              </span>
            )}
          </div>
        ))}
        <div ref={endRef}/>
      </div>
    </div>
  );
};
