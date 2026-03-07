# Chess App

A React + Vite + TypeScript + Tailwind chess application with a built-in minimax AI engine.

## Setup

```bash
npm install
npm run dev
```

## Project Structure

```
src/
├── engine/
│   ├── types.ts       — All TypeScript types (GameState, Move, Piece, etc.)
│   ├── chess.ts       — Full chess rules: move gen, legal moves, check detection, FEN, apply move
│   ├── ai.ts          — Minimax AI with alpha-beta pruning + piece-square tables
│   └── index.ts       — Barrel export
├── hooks/
│   └── useChessGame.ts — React hook: all game state management, player/AI turn logic
├── components/
│   ├── ChessBoard.tsx  — Board rendering, square highlighting, click handling
│   ├── ChessPiece.tsx  — SVG chess pieces
│   ├── MoveHistory.tsx — Scrollable move history sidebar
│   ├── CapturedPieces.tsx — Captured piece display with material score
│   └── PromotionModal.tsx — Pawn promotion picker
├── App.tsx            — Top-level layout + status bar
├── main.tsx
└── index.css
```

## Replacing the AI

The AI lives entirely in `src/engine/ai.ts`. The hook `src/hooks/useChessGame.ts` calls:

```ts
import { getBestMove } from '../engine/ai';
// ...
const result = await getBestMove(currentState, 1500);
const { move } = result;
```

### To plug in your own AI, you have two options:

**Option A — Replace the export in `ai.ts`:**

```ts
// src/engine/ai.ts
import type { GameState, Move } from './types';
import { getAllLegalMoves } from './chess';

export interface AIResult {
  move: Move;
  score: number;
  depth: number;
  nodes: number;
}

export async function getBestMove(state: GameState, timeLimitMs = 1500): Promise<AIResult> {
  // Call your engine here — HTTP, WebSocket, WASM, Worker, etc.
  const response = await fetch('/api/chess-move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fen: toFEN(state) }),
  });
  const data = await response.json();
  // data.move must be a legal Move object for state.turn
  return { move: data.move, score: 0, depth: 0, nodes: 0 };
}
```

**Option B — Use a Web Worker:**

Move the minimax search into a Web Worker to keep the UI responsive at higher depths. The interface in `useChessGame.ts` is already `async`/`await` based, so just return a Promise that resolves when the worker posts back.

## Built-in AI Details

- **Algorithm**: Minimax with alpha-beta pruning
- **Depth**: Iterative deepening from 1→6 within the time budget (default 1500ms)
- **Evaluation**: Material values + piece-square tables (opening/endgame king tables)
- **Move ordering**: MVV-LVA captures first, then quiet moves — dramatically improves pruning
- **Quiescence search**: Extends 4 ply on captures/promotions to avoid the horizon effect

## GameState Shape

```ts
interface GameState {
  board: (Piece | null)[][];  // 8x8, row 0 = rank 8, row 7 = rank 1
  turn: 'w' | 'b';
  castleRights: { wK, wQ, bK, bQ: boolean };
  enPassant: [row, col] | null;
  halfMoveClock: number;
  fullMoveNumber: number;
}
```

## Move Shape

```ts
interface Move {
  from: [row, col];
  to: [row, col];
  piece: Piece;           // e.g. 'wP', 'bQ'
  captured?: Piece;
  promotion?: 'Q'|'R'|'B'|'N';
  castling?: 'K'|'Q';
  enPassant?: boolean;
}
```
