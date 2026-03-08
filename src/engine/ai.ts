/**
 * NNUE Chess AI Engine
 * Loads and uses trained PyTorch model from self-play training
 * Features:
 * - Loads binary .nnue weights from Python training
 * - NNUE evaluation with accumulator updates
 * - Alpha-beta search with move ordering
 * - Quiescence search
 * - Iterative deepening
 */

import type { GameState, Move } from './types';
import { applyMove, getAllLegalMoves, isInCheck, pieceType, pieceColor, getGameResult } from './chess';

// ─── Piece values (for material counting) ─────────────────────────────────────

const PIECE_VALUE: Record<string, number> = {
  P: 100, N: 320, B: 330, R: 500, Q: 900, K: 0,  // King value handled separately
};

// ─── NNUE Evaluator (loads your trained model) ───────────────────────────────

interface NNUEWeights {
  fc1_weight: Float32Array[];  // [256][768] - hidden layer weights
  fc1_bias: Float32Array;      // [256] - hidden layer bias
  fc2_weight: Float32Array;    // [256] - output layer weights
  fc2_bias: number;            // output bias
}

class NNUEEvaluator {
  private weights: NNUEWeights | null = null;
  private accumulator: Float32Array;  // Current hidden layer values
  private pieceIndices: Map<string, number> = new Map([
    ['P', 0], ['N', 1], ['B', 2], ['R', 3], ['Q', 4], ['K', 5],
    ['p', 6], ['n', 7], ['b', 8], ['r', 9], ['q', 10], ['k', 11]
  ]);
  
  constructor() {
    this.accumulator = new Float32Array(256);
  }
  
  /**
   * Load weights from binary .nnue file (exported from Python)
   * Format: "NNUE" magic (4 bytes) + version (4 bytes) + 
   *         for each weight: length (4 bytes) + data (length * 4 bytes)
   */
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
    if (magic !== 'NNUE') {
      throw new Error('Invalid NNUE file format');
    }
    offset += 4;
    
    // Check version
    const version = data.getInt32(offset, true);
    offset += 4;
    console.log(`Loading NNUE v${version}`);
    
    // Load fc1_weight [256][768]
    const fc1_len = data.getInt32(offset, true);
    offset += 4;
    const fc1_weight: Float32Array[] = [];
    const fc1_data = new Float32Array(buffer, offset, fc1_len);
    for (let i = 0; i < 256; i++) {
      fc1_weight[i] = fc1_data.slice(i * 768, (i + 1) * 768);
    }
    offset += fc1_len * 4;
    
    // Load fc1_bias [256]
    const fc1_bias_len = data.getInt32(offset, true);
    offset += 4;
    const fc1_bias = new Float32Array(buffer, offset, fc1_bias_len);
    offset += fc1_bias_len * 4;
    
    // Load fc2_weight [256]
    const fc2_weight_len = data.getInt32(offset, true);
    offset += 4;
    const fc2_weight = new Float32Array(buffer, offset, fc2_weight_len);
    offset += fc2_weight_len * 4;
    
    // Load fc2_bias [1]
    const fc2_bias_len = data.getInt32(offset, true);
    offset += 4;
    const fc2_bias = new Float32Array(buffer, offset, fc2_bias_len)[0];
    
    this.weights = {
      fc1_weight,
      fc1_bias,
      fc2_weight,
      fc2_bias
    };
    
    console.log('NNUE weights loaded successfully');
  }
  
  /**
   * Initialize accumulator for a position
   */
  public setPosition(state: GameState): void {
    if (!this.weights) return;
    
    // Start with bias
    for (let i = 0; i < 256; i++) {
      this.accumulator[i] = this.weights.fc1_bias[i];
    }
    
    // Add contributions from all pieces
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const piece = state.board[r][c];
        if (!piece) continue;
        
        const pieceKey = piece; // e.g., "wP", "bK"
        const pieceIdx = this.pieceIndices.get(pieceKey);
        if (pieceIdx === undefined) continue;
        
        const square = r * 8 + c;
        const featureIdx = pieceIdx * 64 + square;
        
        // Add weights for this feature
        for (let i = 0; i < 256; i++) {
          this.accumulator[i] += this.weights.fc1_weight[i][featureIdx];
        }
      }
    }
  }
  
  /**
   * Incrementally update accumulator after a move
   */
  public updateMove(move: Move): void {
    if (!this.weights) return;
    
    // Remove moved piece from old square
    const pieceIdx = this.pieceIndices.get(move.piece);
    if (pieceIdx !== undefined) {
      const fromSq = move.from.r * 8 + move.from.c;
      const fromFeature = pieceIdx * 64 + fromSq;
      
      for (let i = 0; i < 256; i++) {
        this.accumulator[i] -= this.weights.fc1_weight[i][fromFeature];
      }
      
      // Add moved piece to new square
      const toSq = move.to.r * 8 + move.to.c;
      const toFeature = pieceIdx * 64 + toSq;
      
      for (let i = 0; i < 256; i++) {
        this.accumulator[i] += this.weights.fc1_weight[i][toFeature];
      }
    }
    
    // Remove captured piece
    if (move.captured) {
      const capturedIdx = this.pieceIndices.get(move.captured);
      if (capturedIdx !== undefined) {
        const toSq = move.to.r * 8 + move.to.c;
        const capturedFeature = capturedIdx * 64 + toSq;
        
        for (let i = 0; i < 256; i++) {
          this.accumulator[i] -= this.weights.fc1_weight[i][capturedFeature];
        }
      }
    }
  }
  
  /**
   * Evaluate current position (from perspective of player to move)
   */
  public evaluate(): number {
    if (!this.weights) return 0;
    
    // Clipped ReLU activation
    let output = this.weights.fc2_bias;
    for (let i = 0; i < 256; i++) {
      const hidden = Math.max(0, Math.min(1, this.accumulator[i]));
      output += hidden * this.weights.fc2_weight[i];
    }
    
    // Scale to centipawns and return from current player's perspective
    return output * 100;
  }
}

// ─── Search with NNUE evaluation ─────────────────────────────────────────────

interface SearchResult {
  move: Move;
  score: number;
  depth: number;
  nodes: number;
  pv: Move[];
}

class NNUE_Searcher {
  private evaluator: NNUEEvaluator;
  private nodes = 0;
  private startTime = 0;
  private timeLimit = 0;
  private pvTable: Move[][] = [];
  private killerMoves: Move[][] = [];
  private historyTable: number[][][] = [];
  
  constructor(evaluator: NNUEEvaluator) {
    this.evaluator = evaluator;
    
    // Initialize killer moves (2 per ply)
    for (let i = 0; i < 64; i++) {
      this.killerMoves[i] = [];
    }
    
    // Initialize history table [piece][from][to]
    for (let p = 0; p < 12; p++) {
      this.historyTable[p] = [];
      for (let f = 0; f < 64; f++) {
        this.historyTable[p][f] = new Array(64).fill(0);
      }
    }
  }
  
  public search(state: GameState, timeLimitMs: number): SearchResult {
    this.startTime = Date.now();
    this.timeLimit = timeLimitMs;
    this.nodes = 0;
    this.pvTable = [];
    
    // Set position in evaluator
    this.evaluator.setPosition(state);
    
    const moves = getAllLegalMoves(state);
    if (moves.length === 0) {
      throw new Error('No legal moves');
    }
    
    let bestMove = moves[0];
    let bestScore = -Infinity;
    let pv: Move[] = [];
    
    // Iterative deepening
    for (let depth = 1; depth <= 20; depth++) {
      // Check time
      if (Date.now() - this.startTime > this.timeLimit) break;
      
      // Clear PV for this depth
      this.pvTable[depth] = [];
      
      // Aspiration windows
      let alpha = -Infinity;
      let beta = Infinity;
      let delta = 50;
      
      for (let attempt = 0; attempt < 3; attempt++) {
        if (attempt > 0) {
          alpha = Math.max(-Infinity, bestScore - delta);
          beta = Math.min(Infinity, bestScore + delta);
          delta *= 2;
        }
        
        const score = this.negamax(state, depth, alpha, beta, 0);
        
        if (score <= alpha) {
          // Failed low - research with wider window
          continue;
        }
        if (score >= beta) {
          // Failed high - research with wider window
          continue;
        }
        
        bestScore = score;
        bestMove = this.pvTable[0]?.[0] || bestMove;
        pv = [...(this.pvTable[0] || [])];
        break;
      }
      
      console.log(`Depth ${depth}: score=${bestScore}, nodes=${this.nodes}, pv=${pv.map(m => formatMove(m)).join(' ')}`);
    }
    
    return {
      move: bestMove,
      score: bestScore,
      depth: 0, // Actual depth tracked in search
      nodes: this.nodes,
      pv
    };
  }
  
  private negamax(
    state: GameState,
    depth: number,
    alpha: number,
    beta: number,
    ply: number
  ): number {
    this.nodes++;
    
    // Time check
    if (this.nodes % 1000 === 0) {
      if (Date.now() - this.startTime > this.timeLimit) {
        return 0;
      }
    }
    
    // Check game end
    const result = getGameResult(state);
    if (result !== 'ongoing') {
      if (result === 'checkmate') return -99999 + ply;
      return 0;
    }
    
    // Quiescence search at leaf
    if (depth <= 0) {
      return this.quiescence(state, alpha, beta, ply);
    }
    
    // Null move pruning
    if (depth >= 3 && !isInCheck(state) && this.hasNonPawnMaterial(state)) {
      const R = 2;
      const nullState = { ...state, turn: state.turn === 'w' ? 'b' : 'w' };
      const score = -this.negamax(nullState, depth - 1 - R, -beta, -beta + 1, ply + 1);
      if (score >= beta) return beta;
    }
    
    // Get moves with ordering
    const moves = this.orderMoves(state, ply);
    if (moves.length === 0) {
      return isInCheck(state) ? -99999 + ply : 0;
    }
    
    let bestScore = -Infinity;
    let movesSearched = 0;
    
    for (const move of moves) {
      // Make move
      const next = applyMove(state, move);
      this.evaluator.updateMove(move);
      
      // Late Move Reduction
      let reduction = 0;
      if (depth >= 3 && movesSearched >= 4 && !move.captured && !isInCheck(state)) {
        reduction = 1;
        if (movesSearched >= 8) reduction = 2;
      }
      
      let score: number;
      if (reduction > 0) {
        // Search with reduced depth
        score = -this.negamax(next, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);
        
        // If score beats alpha, research with full depth
        if (score > alpha) {
          score = -this.negamax(next, depth - 1, -alpha - 1, -alpha, ply + 1);
        }
      } else {
        score = -this.negamax(next, depth - 1, -alpha - 1, -alpha, ply + 1);
      }
      
      // PV Search
      if (score > alpha && score < beta && movesSearched > 0) {
        score = -this.negamax(next, depth - 1, -beta, -alpha, ply + 1);
      }
      
      // Undo move (reset accumulator)
      this.evaluator.setPosition(state);
      
      if (score > bestScore) {
        bestScore = score;
        
        // Update PV
        this.pvTable[ply] = [move, ...(this.pvTable[ply + 1] || [])];
        
        // Update killer moves
        if (!move.captured) {
          this.killerMoves[ply][1] = this.killerMoves[ply][0];
          this.killerMoves[ply][0] = move;
        }
      }
      
      if (score > alpha) {
        alpha = score;
        
        // Update history
        if (!move.captured) {
          const pieceIdx = this.getPieceIndex(move.piece);
          const fromSq = move.from.r * 8 + move.from.c;
          const toSq = move.to.r * 8 + move.to.c;
          this.historyTable[pieceIdx][fromSq][toSq] += depth * depth;
        }
      }
      
      if (alpha >= beta) {
        break; // Beta cutoff
      }
      
      movesSearched++;
    }
    
    return bestScore;
  }
  
  private quiescence(
    state: GameState,
    alpha: number,
    beta: number,
    ply: number
  ): number {
    this.nodes++;
    
    // Stand-pat evaluation
    const standPat = this.evaluator.evaluate();
    if (standPat >= beta) return beta;
    if (standPat > alpha) alpha = standPat;
    
    // Only consider captures
    const moves = getAllLegalMoves(state)
      .filter(m => m.captured || m.promotion || isInCheck(applyMove(state, m)))
      .sort((a, b) => {
        const aVal = a.captured ? PIECE_VALUE[a.captured[1]] || 0 : 0;
        const bVal = b.captured ? PIECE_VALUE[b.captured[1]] || 0 : 0;
        return bVal - aVal;
      });
    
    for (const move of moves) {
      const next = applyMove(state, move);
      this.evaluator.updateMove(move);
      const score = -this.quiescence(next, -beta, -alpha, ply + 1);
      this.evaluator.setPosition(state);
      
      if (score >= beta) return beta;
      if (score > alpha) alpha = score;
    }
    
    return alpha;
  }
  
  private orderMoves(state: GameState, ply: number): Move[] {
    const moves = getAllLegalMoves(state);
    const ttMove = this.pvTable[ply]?.[0];
    
    return moves.sort((a, b) => {
      // PV move first
      if (ttMove && this.movesEqual(a, ttMove)) return -1;
      if (ttMove && this.movesEqual(b, ttMove)) return 1;
      
      const aScore = this.scoreMove(state, a, ply);
      const bScore = this.scoreMove(state, b, ply);
      return bScore - aScore;
    });
  }
  
  private scoreMove(state: GameState, move: Move, ply: number): number {
    // Captures (MVV-LVA)
    if (move.captured) {
      const victim = PIECE_VALUE[move.captured[1]] || 0;
      const attacker = PIECE_VALUE[move.piece[1]] || 0;
      return 10000 + victim * 10 - attacker;
    }
    
    // Promotions
    if (move.promotion) {
      return 8000 + PIECE_VALUE[move.promotion] || 0;
    }
    
    // Killer moves
    if (this.killerMoves[ply]?.some(k => k && this.movesEqual(move, k))) {
      return 5000;
    }
    
    // History heuristic
    const pieceIdx = this.getPieceIndex(move.piece);
    const fromSq = move.from.r * 8 + move.from.c;
    const toSq = move.to.r * 8 + move.to.c;
    const history = this.historyTable[pieceIdx]?.[fromSq]?.[toSq] || 0;
    if (history > 0) {
      return 1000 + history;
    }
    
    // Center control
    const toRank = move.to.r;
    const toFile = move.to.c;
    if (toRank >= 3 && toRank <= 4 && toFile >= 3 && toFile <= 4) {
      return 500;
    }
    
    return 0;
  }
  
  private movesEqual(a: Move, b: Move): boolean {
    return a.from.r === b.from.r && a.from.c === b.from.c &&
           a.to.r === b.to.r && a.to.c === b.to.c &&
           a.promotion === b.promotion;
  }
  
  private getPieceIndex(piece: string): number {
    const pieces = ['P', 'N', 'B', 'R', 'Q', 'K'];
    const color = piece[0];
    const pt = piece[1];
    const base = pieces.indexOf(pt);
    return color === 'w' ? base : base + 6;
  }
  
  private hasNonPawnMaterial(state: GameState): boolean {
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const p = state.board[r][c];
        if (p && p[1] !== 'P' && p[1] !== 'K') return true;
      }
    }
    return false;
  }
}

// ─── Main AI Engine ──────────────────────────────────────────────────────────

export class TrainedChessAI {
  private evaluator: NNUEEvaluator;
  private searcher: NNUE_Searcher | null = null;
  private isReady = false;
  
  constructor() {
    this.evaluator = new NNUEEvaluator();
  }
  
  /**
   * Initialize the AI by loading trained weights
   * Call this once when your app starts
   */
  public async initialize(modelUrl: string = '/best_network.nnue'): Promise<void> {
    console.log('Loading trained chess model...');
    await this.evaluator.loadWeights(modelUrl);
    this.searcher = new NNUE_Searcher(this.evaluator);
    this.isReady = true;
    console.log('Chess AI ready!');
  }
  
  /**
   * Get the best move for a position
   */
  public async getBestMove(state: GameState, timeLimitMs: number = 2000): Promise<{
    move: Move;
    score: number;
    depth: number;
    nodes: number;
    pv: string[];
  }> {
    if (!this.isReady || !this.searcher) {
      throw new Error('AI not initialized. Call initialize() first.');
    }
    
    return new Promise(resolve => {
      setTimeout(() => {
        const result = this.searcher!.search(state, timeLimitMs);
        
        resolve({
          move: result.move,
          score: result.score,
          depth: result.depth,
          nodes: result.nodes,
          pv: result.pv.map(m => formatMove(m))
        });
      }, 10);
    });
  }
  
  /**
   * Check if AI is ready
   */
  public ready(): boolean {
    return this.isReady;
  }
}

// ─── Helper Functions ────────────────────────────────────────────────────────

function formatMove(move: Move): string {
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const fromFile = files[move.from.c];
  const fromRank = move.from.r + 1;
  const toFile = files[move.to.c];
  const toRank = move.to.r + 1;
  let str = `${fromFile}${fromRank}${toFile}${toRank}`;
  if (move.promotion) {
    str += move.promotion.toLowerCase();
  }
  return str;
}
