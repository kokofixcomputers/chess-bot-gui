/**
 * Chess AI Engine
 * ---------------
 * Implements a classical minimax search with:
 *  - Alpha-beta pruning (negamax formulation)
 *  - Iterative deepening  
 *  - Transposition table (Zobrist hash based)
 *  - Move ordering (TT hit, check, MVV-LVA, promotion)
 *  - NNUE evaluation from trained model
 *  - Quiescence search to avoid horizon effect
 */

import type { GameState, Move } from './types';
import { applyMove, getAllLegalMoves, isInCheck, pieceType, pieceColor, getGameResult } from './chess';

// ─── Piece values ─────────────────────────────────────────────────────────────

const PIECE_VALUE: Record<string, number> = {
  P: 100, N: 320, B: 330, R: 500, Q: 900, K: 0,
};

// ─── NNUE Evaluation (from trained model) ────────────────────────────────────

interface NNUEWeights {
  fc1_weight: Float32Array[];  // [256][768]
  fc1_bias: Float32Array;      // [256]
  fc2_weight: Float32Array;    // [256]
  fc2_bias: number;
}

class NNUEEvaluator {
  private weights: NNUEWeights | null = null;
  private accumulator: Float32Array = new Float32Array(256);
  private pieceMap: Record<string, number> = {
    'wP': 0, 'wN': 1, 'wB': 2, 'wR': 3, 'wQ': 4, 'wK': 5,
    'bP': 6, 'bN': 7, 'bB': 8, 'bR': 9, 'bQ': 10, 'bK': 11
  };

  public async loadWeights(url: string): Promise<void> {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    const data = new DataView(buffer);
    let offset = 0;
    
    // Check magic number
    const magic = String.fromCharCode(
      data.getUint8(0), data.getUint8(1), 
      data.getUint8(2), data.getUint8(3)
    );
    if (magic !== 'NNUE') throw new Error('Invalid NNUE file');
    offset += 4;
    
    // Version
    const version = data.getInt32(offset, true);
    offset += 4;
    console.log(`Loading NNUE v${version}`);
    
    // Load fc1_weight
    const fc1_len = data.getInt32(offset, true);
    offset += 4;
    const fc1_weight: Float32Array[] = [];
    for (let i = 0; i < 256; i++) {
      const start = offset + i * 768 * 4;
      fc1_weight[i] = new Float32Array(buffer, start, 768);
    }
    offset += fc1_len * 4;
    
    // Load fc1_bias
    const fc1_bias_len = data.getInt32(offset, true);
    offset += 4;
    const fc1_bias = new Float32Array(buffer, offset, fc1_bias_len);
    offset += fc1_bias_len * 4;
    
    // Load fc2_weight
    const fc2_weight_len = data.getInt32(offset, true);
    offset += 4;
    const fc2_weight = new Float32Array(buffer, offset, fc2_weight_len);
    offset += fc2_weight_len * 4;
    
    // Load fc2_bias
    const fc2_bias_len = data.getInt32(offset, true);
    offset += 4;
    const fc2_bias = new Float32Array(buffer, offset, fc2_bias_len)[0];
    
    this.weights = { fc1_weight, fc1_bias, fc2_weight, fc2_bias };
    console.log('NNUE weights loaded');
  }

  public setPosition(state: GameState): void {
    if (!this.weights) return;
    
    // Start with bias
    for (let i = 0; i < 256; i++) {
      this.accumulator[i] = this.weights.fc1_bias[i];
    }
    
    // Add piece contributions
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const piece = state.board[r][c];
        if (!piece) continue;
        
        const pieceIdx = this.pieceMap[piece];
        if (pieceIdx === undefined) continue;
        
        const square = r * 8 + c;
        const featureIdx = pieceIdx * 64 + square;
        
        for (let i = 0; i < 256; i++) {
          this.accumulator[i] += this.weights.fc1_weight[i][featureIdx];
        }
      }
    }
  }

  public evaluate(): number {
    if (!this.weights) return 0;
    
    // Clipped ReLU and output layer
    let output = this.weights.fc2_bias;
    for (let i = 0; i < 256; i++) {
      const hidden = Math.max(0, Math.min(1, this.accumulator[i]));
      output += hidden * this.weights.fc2_weight[i];
    }
    
    return output * 100; // Scale to centipawns
  }
}

// Create singleton evaluator
const evaluator = new NNUEEvaluator();

// ─── Zobrist Hashing ─────────────────────────────────────────────────────────

const ZOBRIST_PIECES = new Uint32Array(12 * 64);
const ZOBRIST_SIDE = BigInt('0x123456789ABCDEF0');

function initZobrist() {
  for (let i = 0; i < ZOBRIST_PIECES.length; i++) {
    ZOBRIST_PIECES[i] = Math.floor(Math.random() * 0x100000000);
  }
}
initZobrist();

function computeZobrist(state: GameState): bigint {
  let hash = 0n;
  
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = state.board[r][c];
      if (p) {
        const pt = pieceType(p);
        const pc = pieceColor(p);
        const pieceIdx = (pc === 'w' ? 0 : 6) + 'PNBRQK'.indexOf(pt);
        const sq = r * 8 + c;
        hash ^= BigInt(ZOBRIST_PIECES[pieceIdx * 64 + sq]);
      }
    }
  }
  
  if (state.turn === 'b') hash ^= ZOBRIST_SIDE;
  return hash;
}

// ─── Transposition Table ─────────────────────────────────────────────────────

interface TTEntry {
  depth: number;
  score: number;
  flag: 0 | 1 | 2;
  move?: Move;
}

const TT_SIZE = 1 << 20;
const TT = new Map<bigint, TTEntry>();

function getTTMove(hash: bigint): Move | undefined {
  return TT.get(hash)?.move;
}

function storeTT(hash: bigint, depth: number, score: number, flag: 0 | 1 | 2, move?: Move) {
  if (!TT.has(hash) || depth >= (TT.get(hash)?.depth ?? 0)) {
    TT.set(hash, { depth, score, flag, move });
    if (TT.size > TT_SIZE) {
      const firstKey = TT.keys().next().value;
      TT.delete(firstKey);
    }
  }
}

// ─── Evaluation (now using NNUE) ─────────────────────────────────────────────

export function evaluate(state: GameState): number {
  const result = getGameResult(state);
  if (result === 'checkmate') return state.turn === 'w' ? -99999 : 99999;
  if (result === 'stalemate' || result === 'draw-50') return 0;

  evaluator.setPosition(state);
  return evaluator.evaluate();
}

// ─── Move ordering ────────────────────────────────────────────────────────────

function scoreMoveForOrdering(state: GameState, move: Move, ttMove?: Move): number {
  if (ttMove && areMovesEqual(move, ttMove)) return 1000000;
  
  if (isInCheck(applyMove(state, move))) return 50000;
  
  if (move.captured) {
    const attacker = PIECE_VALUE[move.piece[1]] ?? 0;
    const victim = PIECE_VALUE[move.captured[1]] ?? 0;
    return 10000 + (victim * 10 - attacker);
  }
  
  if (move.promotion) return 20000 + PIECE_VALUE[move.promotion ?? 'Q'];
  
  return 0;
}

function areMovesEqual(a: Move, b: Move): boolean {
  return a.from.r === b.from.r && a.from.c === b.from.c &&
         a.to.r === b.to.r && a.to.c === b.to.c &&
         a.promotion === b.promotion;
}

function orderMoves(state: GameState, moves: Move[], ttMove?: Move): Move[] {
  return [...moves].sort((a, b) => 
    scoreMoveForOrdering(state, b, ttMove) - scoreMoveForOrdering(state, a, ttMove)
  );
}

// ─── Quiescence search ────────────────────────────────────────────────────────

function quiescence(state: GameState, alpha: number, beta: number): number {
  const stand = evaluate(state);
  if (stand >= beta) return beta;
  if (stand > alpha) alpha = stand;

  const forcingMoves = getAllLegalMoves(state).filter(m => 
    m.captured || isInCheck(applyMove(state, m))
  );
  
  for (const move of orderMoves(state, forcingMoves)) {
    const next = applyMove(state, move);
    const score = -quiescence(next, -beta, -alpha);
    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }
  return alpha;
}

// ─── Negamax with alpha-beta pruning ─────────────────────────────────────────

let nodesSearched = 0;

function negamax(state: GameState, depth: number, alpha: number, beta: number): number {
  nodesSearched++;
  
  const hash = computeZobrist(state);
  const ttEntry = TT.get(hash);
  
  if (ttEntry && ttEntry.depth >= depth) {
    if (ttEntry.flag === 0) return ttEntry.score;
    if (ttEntry.flag === 1 && ttEntry.score >= beta) return ttEntry.score;
    if (ttEntry.flag === 2 && ttEntry.score <= alpha) return ttEntry.score;
  }
  
  const ttMove = ttEntry?.move;
  
  const result = getGameResult(state);
  if (result !== 'ongoing') {
    if (result === 'checkmate') return -99999;
    return 0;
  }
  
  if (depth === 0) return quiescence(state, alpha, beta);
  
  let bestScore = -Infinity;
  
  const moves = orderMoves(state, getAllLegalMoves(state), ttMove);
  for (const move of moves) {
    const next = applyMove(state, move);
    const score = -negamax(next, depth - 1, -beta, -alpha);
    
    if (score > bestScore) bestScore = score;
    if (bestScore > alpha) alpha = bestScore;
    if (alpha >= beta) break;
  }
  
  let flag = 0;
  if (bestScore <= alpha) flag = 2;
  else if (bestScore >= beta) flag = 1;
  
  storeTT(hash, depth, bestScore, flag, moves[0]);
  
  return bestScore;
}

// ─── Public API ───────────────────────────────────────────────────────────────

export interface AIResult {
  move: Move;
  score: number;
  depth: number;
  nodes: number;
}

export async function getBestMove(state: GameState, timeLimitMs = 1500): Promise<AIResult> {
  return new Promise(resolve => {
    setTimeout(() => {
      const start = Date.now();
      nodesSearched = 0;
      
      const hash = computeZobrist(state);
      TT.clear();
      
      const moves = getAllLegalMoves(state);
      if (moves.length === 0) throw new Error('No legal moves');
      if (moves.length === 1) {
        resolve({ move: moves[0], score: 0, depth: 0, nodes: 1 });
        return;
      }
      
      let bestMove = moves[0];
      let bestScore = -Infinity;
      let completedDepth = 0;
      
      for (let depth = 1; depth <= 8; depth++) {
        if (Date.now() - start > timeLimitMs) break;
        
        let depthBestMove = moves[0];
        let depthBestScore = -Infinity;
        let alpha = -Infinity;
        let beta = Infinity;
        
        const ttMove = getTTMove(hash);
        const ordered = orderMoves(state, moves, ttMove);
        
        for (const move of ordered) {
          if (Date.now() - start > timeLimitMs) break;
          
          const next = applyMove(state, move);
          const score = -negamax(next, depth - 1, -beta, -alpha);
          
          if (score > depthBestScore) {
            depthBestScore = score;
            depthBestMove = move;
          }
          
          if (score > alpha) alpha = score;
        }
        
        if (depthBestMove) {
          bestMove = depthBestMove;
          bestScore = depthBestScore;
          completedDepth = depth;
        }
      }
      
      resolve({ move: bestMove, score: bestScore, depth: completedDepth, nodes: nodesSearched });
    }, 10);
  });
}

// ─── Initialize the AI with your trained model ───────────────────────────────

let initialized = false;

export async function initAI(modelUrl: string = '/best_network.nnue'): Promise<void> {
  if (initialized) return;
  await evaluator.loadWeights(modelUrl);
  initialized = true;
  console.log('AI initialized with trained model');
}
