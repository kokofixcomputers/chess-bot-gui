import React from 'react';
import type { Piece } from '../engine/types';

const SYMBOL: Record<string, string> = {
  wK:'♔',wQ:'♕',wR:'♖',wB:'♗',wN:'♘',wP:'♙',
  bK:'♚',bQ:'♛',bR:'♜',bB:'♝',bN:'♞',bP:'♟',
};
const VALUE: Record<string, number> = { P:1, N:3, B:3, R:5, Q:9, K:0 };

interface Props {
  pieces: Piece[];
  label: string;
}

export const CapturedPieces: React.FC<Props> = ({ pieces, label }) => {
  const sorted = [...pieces].sort((a, b) => VALUE[b[1]] - VALUE[a[1]]);
  const score = pieces.reduce((s, p) => s + VALUE[p[1]], 0);

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, height: 28 }}>
      <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.25)', minWidth: 20, fontFamily: 'DM Mono, monospace' }}>{label}</span>
      <div style={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        {sorted.map((p, i) => (
          <span key={i} style={{ fontSize: 17, opacity: 0.75 }}>{SYMBOL[p]}</span>
        ))}
      </div>
      {score > 0 && (
        <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', fontFamily: 'DM Mono, monospace' }}>+{score}</span>
      )}
    </div>
  );
};
