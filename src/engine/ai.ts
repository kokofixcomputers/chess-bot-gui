/**
 * Chess AI Engine
 * ---------------
 * Implements a classical minimax search with:
 *  - Alpha-beta pruning
 *  - Iterative deepening
 *  - Move ordering (captures first, then quiet moves scored by piece-square tables)
 *  - Piece-square tables for positional evaluation
 *  - Quiescence search to avoid horizon effect
 *
 * To replace with your own AI: export a function matching the signature:
 *   getBestMove(state: GameState, depthMs?: number): Promise<Move>
 */

import type { GameState, Move } from './types';
import { applyMove, getAllLegalMoves, isInCheck, pieceType, pieceColor, getGameResult } from './chess';

// ─── Piece values ─────────────────────────────────────────────────────────────

const PIECE_VALUE: Record<string, number> = {
  P: 100, N: 320, B: 330, R: 500, Q: 900, K: 20000,
};

// ─── Piece-Square Tables (white's perspective; rows 0=rank8, 7=rank1) ─────────

const PST: Record<string, number[][]> = {
  P: [
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [ 5,  5, 10, 25, 25, 10,  5,  5],
    [ 0,  0,  0, 20, 20,  0,  0,  0],
    [ 5, -5,-10,  0,  0,-10, -5,  5],
    [ 5, 10, 10,-20,-20, 10, 10,  5],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
  ],
  N: [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50],
  ],
  B: [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20],
  ],
  R: [
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 5, 10, 10, 10, 10, 10, 10,  5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [ 0,  0,  0,  5,  5,  0,  0,  0],
  ],
  Q: [
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [ -5,  0,  5,  5,  5,  5,  0, -5],
    [  0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20],
  ],
  K_MG: [
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [ 20, 20,  0,  0,  0,  0, 20, 20],
    [ 20, 30, 10,  0,  0, 10, 30, 20],
  ],
  K_EG: [
    [-50,-40,-30,-20,-20,-30,-40,-50],
    [-30,-20,-10,  0,  0,-10,-20,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-30,  0,  0,  0,  0,-30,-30],
    [-50,-30,-30,-30,-30,-30,-30,-50],
  ],
};

function getPST(pt: string, color: 'w' | 'b', r: number, c: number, isEndgame: boolean): number {
  const table = pt === 'K' ? (isEndgame ? PST.K_EG : PST.K_MG) : PST[pt];
  if (!table) return 0;
  const row = color === 'w' ? r : 7 - r;
  return table[row][c];
}

// ─── Evaluation ───────────────────────────────────────────────────────────────

function isEndgame(state: GameState): boolean {
  let queens = 0, minors = 0;
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
    const p = state.board[r][c];
    if (!p) continue;
    if (p[1] === 'Q') queens++;
    if (p[1] === 'N' || p[1] === 'B') minors++;
  }
  return queens === 0 || (queens === 2 && minors <= 2);
}

export function evaluate(state: GameState): number {
  const result = getGameResult(state);
  if (result === 'checkmate') return state.turn === 'w' ? -99999 : 99999;
  if (result === 'stalemate' || result === 'draw-50') return 0;

  const eg = isEndgame(state);
  let score = 0;

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = state.board[r][c];
      if (!p) continue;
      const color = p[0] as 'w' | 'b';
      const pt = p[1];
      const val = (PIECE_VALUE[pt] ?? 0) + getPST(pt, color, r, c, eg);
      score += color === 'w' ? val : -val;
    }
  }

  return score;
}

// ─── Move ordering ────────────────────────────────────────────────────────────

function scoreMoveForOrdering(move: Move): number {
  let score = 0;
  if (move.captured) {
    const attacker = PIECE_VALUE[move.piece[1]] ?? 0;
    const victim = PIECE_VALUE[move.captured[1]] ?? 0;
    score += 10 * victim - attacker + 10000; // MVV-LVA
  }
  if (move.promotion) score += PIECE_VALUE[move.promotion] + 5000;
  if (move.castling) score += 500;
  return score;
}

function orderMoves(moves: Move[]): Move[] {
  return [...moves].sort((a, b) => scoreMoveForOrdering(b) - scoreMoveForOrdering(a));
}

// ─── Quiescence search ────────────────────────────────────────────────────────

function quiescence(state: GameState, alpha: number, beta: number, maxDepth: number): number {
  const stand = evaluate(state);
  if (maxDepth <= 0) return stand;
  if (stand >= beta) return beta;
  if (stand > alpha) alpha = stand;

  const captures = getAllLegalMoves(state).filter(m => m.captured || m.promotion);
  for (const move of orderMoves(captures)) {
    const next = applyMove(state, move);
    const score = -quiescence(next, -beta, -alpha, maxDepth - 1);
    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }
  return alpha;
}

// ─── Minimax with alpha-beta ──────────────────────────────────────────────────

let nodesSearched = 0;

function alphaBeta(state: GameState, depth: number, alpha: number, beta: number, maximizing: boolean): number {
  nodesSearched++;

  if (depth === 0) return quiescence(state, alpha, beta, 4);

  const result = getGameResult(state);
  if (result !== 'ongoing') {
    if (result === 'checkmate') return maximizing ? -99999 - depth : 99999 + depth;
    return 0;
  }

  const moves = orderMoves(getAllLegalMoves(state));

  if (maximizing) {
    let best = -Infinity;
    for (const move of moves) {
      const next = applyMove(state, move);
      const score = alphaBeta(next, depth - 1, alpha, beta, false);
      best = Math.max(best, score);
      alpha = Math.max(alpha, score);
      if (alpha >= beta) break;
    }
    return best;
  } else {
    let best = Infinity;
    for (const move of moves) {
      const next = applyMove(state, move);
      const score = alphaBeta(next, depth - 1, alpha, beta, true);
      best = Math.min(best, score);
      beta = Math.min(beta, score);
      if (alpha >= beta) break;
    }
    return best;
  }
}

// ─── Public API ───────────────────────────────────────────────────────────────

export interface AIResult {
  move: Move;
  score: number;
  depth: number;
  nodes: number;
}

/**
 * Get the best move for the current player using iterative deepening.
 * @param state  The current game state
 * @param timeLimitMs  How long to think (default 1500ms)
 * @returns The best move found
 *
 * ── HOW TO REPLACE WITH YOUR OWN AI ──────────────────────────────────────────
 * Export a function with this signature from your own module, then update the
 * import in src/hooks/useChessGame.ts:
 *
 *   export async function getBestMove(state: GameState, timeLimitMs?: number): Promise<AIResult>
 *
 * Your function receives the full GameState and should return an AIResult with
 * at minimum a valid `move` field. The move must be a legal move for state.turn.
 * ─────────────────────────────────────────────────────────────────────────────
 */
export async function getBestMove(state: GameState, timeLimitMs = 1500): Promise<AIResult> {
  return new Promise(resolve => {
    // Use setTimeout to not block the main thread
    setTimeout(() => {
      const start = Date.now();
      nodesSearched = 0;

      const moves = orderMoves(getAllLegalMoves(state));
      if (moves.length === 0) throw new Error('No legal moves');
      if (moves.length === 1) {
        resolve({ move: moves[0], score: 0, depth: 0, nodes: 1 });
        return;
      }

      const isMax = state.turn === 'w';
      let bestMove = moves[0];
      let bestScore = isMax ? -Infinity : Infinity;
      let completedDepth = 0;

      // Iterative deepening: increase depth until time runs out
      for (let depth = 1; depth <= 6; depth++) {
        let depthBest = moves[0];
        let depthBestScore = isMax ? -Infinity : Infinity;

        for (const move of moves) {
          const next = applyMove(state, move);
          const score = alphaBeta(next, depth - 1, -Infinity, Infinity, !isMax);

          if (isMax ? score > depthBestScore : score < depthBestScore) {
            depthBestScore = score;
            depthBest = move;
          }
        }

        bestMove = depthBest;
        bestScore = depthBestScore;
        completedDepth = depth;

        if (Date.now() - start > timeLimitMs) break;
      }

      resolve({ move: bestMove, score: bestScore, depth: completedDepth, nodes: nodesSearched });
    }, 10);
  });
}
