import React from 'react';
import { useChessGame } from './hooks/useChessGame';
import { ChessBoard } from './components/ChessBoard';
import { MoveHistory } from './components/MoveHistory';
import { CapturedPieces } from './components/CapturedPieces';
import { PromotionModal } from './components/PromotionModal';

export default function App() {
  const game = useChessGame();

  const {
    gameState, selected, legalMovesForSelected, phase,
    history, capturedByWhite, capturedByBlack, lastMove,
    aiThinking, aiDepth, aiNodes, inCheck,
    promotionPending, shakeSq,
    selectSquare, completePromotion, resetGame, undoMove,
  } = game;

  const statusText = () => {
    if (phase === 'checkmate') return gameState.turn === 'w' ? '🏴 AI wins by checkmate' : '🎉 You win by checkmate!';
    if (phase === 'stalemate') return '½ Stalemate — Draw';
    if (phase === 'draw-50') return '½ 50-move rule — Draw';
    if (aiThinking) return 'AI is thinking…';
    if (inCheck && gameState.turn === 'w') return '⚠ You are in check!';
    if (inCheck && gameState.turn === 'b') return '⚠ AI is in check!';
    return gameState.turn === 'w' ? 'Your turn' : "AI's turn";
  };

  const isGameOver = phase !== 'playing';
  const statusColor = phase === 'checkmate' && gameState.turn === 'b'
    ? 'rgba(34,197,94,0.3)'
    : phase === 'checkmate'
      ? 'rgba(239,68,68,0.25)'
      : inCheck
        ? 'rgba(239,68,68,0.2)'
        : aiThinking
          ? 'rgba(168,85,247,0.2)'
          : 'rgba(255,255,255,0.05)';

  const statusBorder = phase === 'checkmate' && gameState.turn === 'b'
    ? 'rgba(34,197,94,0.4)'
    : phase === 'checkmate'
      ? 'rgba(239,68,68,0.4)'
      : inCheck
        ? 'rgba(239,68,68,0.35)'
        : aiThinking
          ? 'rgba(168,85,247,0.4)'
          : 'rgba(255,255,255,0.1)';

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(145deg, #0d0f1a 0%, #131525 50%, #0f1220 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 24,
    }}>
      {/* Ambient glows */}
      <div style={{ position: 'fixed', top: '5%', left: '8%', width: 400, height: 400, borderRadius: '50%', background: 'radial-gradient(circle, rgba(99,102,241,0.07) 0%, transparent 70%)', pointerEvents: 'none' }}/>
      <div style={{ position: 'fixed', bottom: '10%', right: '5%', width: 500, height: 500, borderRadius: '50%', background: 'radial-gradient(circle, rgba(168,85,247,0.05) 0%, transparent 70%)', pointerEvents: 'none' }}/>

      <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start', width: '100%', maxWidth: 1060 }}>

        {/* Left — Board area */}
        <div style={{ flex: 'none', display: 'flex', flexDirection: 'column', gap: 10 }}>

          {/* Status bar */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
            <div style={{
              background: statusColor,
              border: `1px solid ${statusBorder}`,
              borderRadius: 100,
              padding: '7px 18px',
              fontSize: 13.5,
              fontWeight: 500,
              color: '#e2e8f0',
              backdropFilter: 'blur(10px)',
              display: 'flex', alignItems: 'center', gap: 8,
              transition: 'all 0.3s',
            }}>
              {aiThinking && (
                <span className="animate-pulse-dot" style={{ display: 'inline-block', width: 7, height: 7, borderRadius: '50%', background: '#a855f7' }}/>
              )}
              {statusText()}
            </div>

            <div style={{ display: 'flex', gap: 8 }}>
              {history.length >= 2 && !isGameOver && (
                <button onClick={undoMove} style={btnStyle('#334155', 'rgba(255,255,255,0.1)')}>
                  ↩ Undo
                </button>
              )}
              <button onClick={resetGame} style={btnStyle('#1e293b', 'rgba(255,255,255,0.08)')}>
                New Game
              </button>
            </div>
          </div>

          {/* Captured by white (black pieces) */}
          <CapturedPieces pieces={capturedByBlack} label="W:" />

          {/* Board */}
          <ChessBoard
            gameState={gameState}
            selected={selected}
            legalMoves={legalMovesForSelected}
            lastMove={lastMove}
            shakeSq={shakeSq}
            inCheck={inCheck}
            onSquareClick={selectSquare}
          />

          {/* Captured by black (white pieces) */}
          <CapturedPieces pieces={capturedByWhite} label="B:" />
        </div>

        {/* Right — Side panel */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 14, alignSelf: 'stretch', minWidth: 220, maxWidth: 280 }}>

          {/* Players */}
          <PlayerCard label="AI (Black)" icon="🤖" active={gameState.turn === 'b' && !isGameOver} thinking={aiThinking} depth={aiDepth} nodes={aiNodes} />
          <PlayerCard label="You (White)" icon="👤" active={gameState.turn === 'w' && !isGameOver} thinking={false} />

          {/* Move history */}
          <MoveHistory history={history} />

          {/* Legend */}
          <div style={{
            background: 'rgba(255,255,255,0.02)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 12,
            padding: '11px 14px',
            fontSize: 12, color: 'rgba(255,255,255,0.25)', lineHeight: 1.75,
          }}>
            <div style={{ fontWeight: 700, fontSize: 10, letterSpacing: '0.09em', textTransform: 'uppercase', marginBottom: 6, color: 'rgba(255,255,255,0.3)' }}>Controls</div>
            Click piece → see legal moves<br/>
            Click dot → make move<br/>
            Click piece again → deselect
          </div>
        </div>
      </div>

      {/* Promotion modal */}
      {promotionPending && (
        <PromotionModal color="w" onSelect={completePromotion} />
      )}
    </div>
  );
}

// ─── PlayerCard ───────────────────────────────────────────────────────────────

function PlayerCard({ label, icon, active, thinking, depth, nodes }: {
  label: string; icon: string; active: boolean; thinking: boolean; depth?: number; nodes?: number;
}) {
  return (
    <div style={{
      background: active
        ? 'linear-gradient(135deg, rgba(99,102,241,0.2), rgba(168,85,247,0.15))'
        : 'rgba(255,255,255,0.03)',
      border: `1px solid ${active ? 'rgba(99,102,241,0.45)' : 'rgba(255,255,255,0.07)'}`,
      borderRadius: 14,
      padding: '13px 16px',
      display: 'flex', alignItems: 'center', gap: 12,
      transition: 'all 0.3s',
    }}>
      <span style={{ fontSize: 22 }}>{icon}</span>
      <div style={{ flex: 1 }}>
        <div style={{ color: active ? '#c4b5fd' : '#64748b', fontWeight: 600, fontSize: 14 }}>{label}</div>
        <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.25)', marginTop: 2 }}>
          {thinking ? 'Thinking…' : active ? 'Active' : 'Waiting'}
          {depth && depth > 0 && !thinking && <span style={{ marginLeft: 6, fontFamily: 'DM Mono, monospace' }}>d{depth} {nodes ? `· ${(nodes / 1000).toFixed(1)}k` : ''}</span>}
        </div>
      </div>
      {active && (
        <div className={thinking ? 'animate-pulse-dot' : ''} style={{
          width: 8, height: 8, borderRadius: '50%',
          background: active ? '#818cf8' : 'transparent',
          boxShadow: active ? '0 0 10px #6366f1' : 'none',
        }}/>
      )}
    </div>
  );
}

function btnStyle(bg: string, border: string) {
  return {
    background: bg,
    border: `1px solid ${border}`,
    borderRadius: 100,
    padding: '7px 16px',
    color: '#94a3b8',
    fontSize: 12.5,
    cursor: 'pointer',
    fontFamily: 'DM Sans, system-ui, sans-serif',
    fontWeight: 500,
    transition: 'all 0.2s',
  } as React.CSSProperties;
}
