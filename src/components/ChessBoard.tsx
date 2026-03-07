import React from 'react';
import type { GameState, Move, Square, Piece } from '../engine/types';
import { pieceColor } from '../engine/chess';
import { ChessPiece } from './ChessPiece';

interface Props {
  gameState: GameState;
  selected: Square | null;
  legalMoves: Move[];
  lastMove: Move | null;
  shakeSq: string | null;
  inCheck: boolean;
  onSquareClick: (sq: Square) => void;
}

const FILES = ['a','b','c','d','e','f','g','h'];

export const ChessBoard: React.FC<Props> = ({
  gameState,
  selected,
  legalMoves,
  lastMove,
  shakeSq,
  inCheck,
  onSquareClick,
}) => {
  const { board, turn } = gameState;

  const isSelected = (r: number, c: number) =>
    selected !== null && selected[0] === r && selected[1] === c;

  const isLegalTarget = (r: number, c: number) =>
    legalMoves.some(m => m.to[0] === r && m.to[1] === c);

  const isLastMoveSquare = (r: number, c: number) =>
    lastMove !== null && (
      (lastMove.from[0] === r && lastMove.from[1] === c) ||
      (lastMove.to[0] === r && lastMove.to[1] === c)
    );

  const isLight = (r: number, c: number) => (r + c) % 2 === 0;

  const isKingInCheck = (r: number, c: number) =>
    inCheck && board[r][c] === `${turn}K`;

  const getSquareBg = (r: number, c: number): string => {
    const light = isLight(r, c);
    if (isKingInCheck(r, c)) return light ? '#fca5a5' : '#ef4444';
    if (isSelected(r, c)) return light ? '#93c5fd' : '#3b82f6';
    if (isLastMoveSquare(r, c)) return light ? '#fef08a' : '#ca8a04';
    return light ? '#dde6f5' : '#9daabf';
  };

  const squareSize = 72;

  return (
    <div style={{ userSelect: 'none' }}>
      {/* File labels top */}
      <div style={{ display: 'flex', paddingLeft: 28, marginBottom: 4 }}>
        {FILES.map(f => (
          <div key={f} style={{ width: squareSize, textAlign: 'center', fontSize: 11, color: 'rgba(255,255,255,0.28)', fontWeight: 700, letterSpacing: '0.08em', fontFamily: 'DM Mono, monospace' }}>
            {f}
          </div>
        ))}
      </div>

      <div style={{ display: 'flex' }}>
        {/* Rank labels */}
        <div style={{ display: 'flex', flexDirection: 'column', width: 28 }}>
          {[8,7,6,5,4,3,2,1].map(n => (
            <div key={n} style={{ height: squareSize, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, color: 'rgba(255,255,255,0.28)', fontWeight: 700, fontFamily: 'DM Mono, monospace' }}>
              {n}
            </div>
          ))}
        </div>

        {/* Board */}
        <div style={{
          borderRadius: 14,
          overflow: 'hidden',
          boxShadow: '0 24px 60px rgba(0,0,0,0.5), 0 0 0 1.5px rgba(255,255,255,0.07)',
          display: 'grid',
          gridTemplateColumns: `repeat(8, ${squareSize}px)`,
          gridTemplateRows: `repeat(8, ${squareSize}px)`,
        }}>
          {board.map((row, r) =>
            row.map((piece, c) => {
              const legalTarget = isLegalTarget(r, c);
              const hasPiece = !!piece;
              const isOwnPiece = piece && pieceColor(piece) === 'w' && turn === 'w';
              const sqKey = `${r}-${c}`;
              const shaking = shakeSq === sqKey;
              const sel = isSelected(r, c);

              return (
                <div
                  key={sqKey}
                  onClick={() => onSquareClick([r, c])}
                  className={shaking ? 'animate-shake' : ''}
                  style={{
                    width: squareSize,
                    height: squareSize,
                    background: getSquareBg(r, c),
                    position: 'relative',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: isOwnPiece || (selected && legalTarget) ? 'pointer' : 'default',
                    transition: 'background 0.12s',
                    boxShadow: sel ? `inset 0 0 0 3px rgba(59,130,246,0.9)` : undefined,
                  }}
                >
                  {/* Legal move dot / ring */}
                  {legalTarget && (
                    hasPiece
                      ? <div style={{
                          position: 'absolute', inset: 0,
                          border: '3.5px solid rgba(99,102,241,0.75)',
                          borderRadius: 0,
                          pointerEvents: 'none',
                          zIndex: 2,
                        }}/>
                      : <div style={{
                          width: 22, height: 22, borderRadius: '50%',
                          background: 'rgba(79,70,229,0.55)',
                          boxShadow: '0 0 14px rgba(99,102,241,0.7)',
                          position: 'absolute', zIndex: 2,
                        }}/>
                  )}

                  {/* Piece */}
                  {piece && (
                    <div style={{
                      position: 'relative', zIndex: 3,
                      transform: sel ? 'scale(1.14) translateY(-4px)' : 'scale(1)',
                      transition: 'transform 0.14s cubic-bezier(0.34,1.56,0.64,1)',
                    }}>
                      <ChessPiece piece={piece} size={squareSize - 12} lifted={sel} />
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* File labels bottom */}
      <div style={{ display: 'flex', paddingLeft: 28, marginTop: 4 }}>
        {FILES.map(f => (
          <div key={f} style={{ width: squareSize, textAlign: 'center', fontSize: 11, color: 'rgba(255,255,255,0.28)', fontWeight: 700, letterSpacing: '0.08em', fontFamily: 'DM Mono, monospace' }}>
            {f}
          </div>
        ))}
      </div>
    </div>
  );
};
