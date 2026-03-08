import { useState, useCallback, useRef, useEffect } from 'react';
import type { GameState, Move, Piece, Square, Color } from '../engine/types';
import { parseFEN, INITIAL_FEN, getLegalMoves, applyMove, getGameResult, isInCheck, moveToAlg, squareToAlg, pieceColor, pieceType } from '../engine/chess';
import { getBestMove, initAI } from '../engine/ai';

export type GamePhase = 'playing' | 'checkmate' | 'stalemate' | 'draw-50' | 'draw-repetition';

export interface HistoryEntry {
  move: Move;
  alg: string;
  fen: string;
  capturedPiece?: Piece;
  aiDepth?: number;
  aiNodes?: number;
  aiScore?: number;
}

export interface ChessGameState {
  gameState: GameState;
  selected: Square | null;
  legalMovesForSelected: Move[];
  phase: GamePhase;
  history: HistoryEntry[];
  capturedByWhite: Piece[];
  capturedByBlack: Piece[];
  lastMove: Move | null;
  aiThinking: boolean;
  aiDepth: number;
  aiNodes: number;
  aiScore: number;
  inCheck: boolean;
  promotionPending: { from: Square; to: Square } | null;
  shakeSq: string | null;
  aiInitialized: boolean;
  aiInitError: string | null;
}

export interface ChessGameActions {
  selectSquare: (sq: Square) => void;
  completePromotion: (piece: 'Q' | 'R' | 'B' | 'N') => void;
  resetGame: () => void;
  undoMove: () => void;
  setAIDifficulty: (timeMs: number) => void;
}

function getPhase(state: GameState): GamePhase {
  const r = getGameResult(state);
  if (r === 'checkmate') return 'checkmate';
  if (r === 'stalemate') return 'stalemate';
  if (r === 'draw-50') return 'draw-50';
  return 'playing';
}

const PLAYER_COLOR: Color = 'w';
const AI_COLOR: Color = 'b';
const DEFAULT_AI_TIME = 1500; // ms

// Hardcoded path to your trained model - CHANGE THIS TO YOUR ACTUAL PATH
const TRAINED_MODEL_PATH = '/best_network.nnue'; // Place your .nnue file in public folder

export function useChessGame(): ChessGameState & ChessGameActions {
  const [gameState, setGameState] = useState<GameState>(() => parseFEN(INITIAL_FEN));
  const [selected, setSelected] = useState<Square | null>(null);
  const [legalMovesForSelected, setLegalMovesForSelected] = useState<Move[]>([]);
  const [phase, setPhase] = useState<GamePhase>('playing');
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [capturedByWhite, setCapturedByWhite] = useState<Piece[]>([]);
  const [capturedByBlack, setCapturedByBlack] = useState<Piece[]>([]);
  const [lastMove, setLastMove] = useState<Move | null>(null);
  const [aiThinking, setAiThinking] = useState(false);
  const [aiDepth, setAiDepth] = useState(0);
  const [aiNodes, setAiNodes] = useState(0);
  const [aiScore, setAiScore] = useState(0);
  const [promotionPending, setPromotionPending] = useState<{ from: Square; to: Square } | null>(null);
  const [shakeSq, setShakeSq] = useState<string | null>(null);
  const [aiInitialized, setAiInitialized] = useState(false);
  const [aiInitError, setAiInitError] = useState<string | null>(null);
  const [aiTimeMs, setAiTimeMs] = useState(DEFAULT_AI_TIME);

  const stateHistoryRef = useRef<GameState[]>([parseFEN(INITIAL_FEN)]);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Initialize AI on mount with hardcoded path
  useEffect(() => {
    const loadAI = async () => {
      try {
        console.log('Loading trained chess model from', TRAINED_MODEL_PATH);
        await initAI(TRAINED_MODEL_PATH);
        setAiInitialized(true);
        setAiInitError(null);
        console.log('AI initialized successfully with trained model!');
      } catch (error) {
        console.error('Failed to initialize AI:', error);
        setAiInitError(error instanceof Error ? error.message : 'Failed to load AI model');
      }
    };
    loadAI();
  }, []);

  const triggerShake = (sq: Square) => {
    const key = `${sq[0]}-${sq[1]}`;
    setShakeSq(key);
    setTimeout(() => setShakeSq(null), 450);
  };

  const triggerAI = useCallback(async (currentState: GameState) => {
    if (!aiInitialized) {
      console.warn('AI not initialized yet, using fallback?');
    }

    setAiThinking(true);
    
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    try {
      const result = await getBestMove(currentState, aiTimeMs);
      const { move, score, depth, nodes } = result;

      const nextState = applyMove(currentState, move);
      const nextPhase = getPhase(nextState);

      stateHistoryRef.current.push(nextState);

      setGameState(nextState);
      setPhase(nextPhase);
      setLastMove(move);
      setAiDepth(depth);
      setAiNodes(nodes);
      setAiScore(score);

      if (move.captured) {
        setCapturedByBlack(prev => [...prev, move.captured!]);
      }

      setHistory(prev => [...prev, {
        move,
        alg: moveToAlg(move),
        fen: '',
        capturedPiece: move.captured,
        aiDepth: depth,
        aiNodes: nodes,
        aiScore: score,
      }]);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        console.log('AI search cancelled');
      } else {
        console.error('AI error:', err);
      }
    } finally {
      setAiThinking(false);
    }
  }, [aiTimeMs, aiInitialized]);

  const executePlayerMove = useCallback((move: Move, currentState: GameState) => {
    const nextState = applyMove(currentState, move);
    const nextPhase = getPhase(nextState);

    stateHistoryRef.current.push(nextState);

    setGameState(nextState);
    setPhase(nextPhase);
    setLastMove(move);
    setSelected(null);
    setLegalMovesForSelected([]);

    if (move.captured) {
      setCapturedByWhite(prev => [...prev, move.captured!]);
    }

    setHistory(prev => [...prev, {
      move,
      alg: moveToAlg(move),
      fen: '',
      capturedPiece: move.captured,
    }]);

    if (nextPhase === 'playing') {
      triggerAI(nextState);
    }
  }, [triggerAI]);

  const selectSquare = useCallback((sq: Square) => {
    if (phase !== 'playing' || aiThinking || promotionPending) return;
    if (gameState.turn !== PLAYER_COLOR) return;

    const [r, c] = sq;
    const piece = gameState.board[r][c];
    const pc = pieceColor(piece);

    if (selected) {
      const [sr, sc] = selected;

      if (sr === r && sc === c) {
        setSelected(null);
        setLegalMovesForSelected([]);
        return;
      }

      const moveCandidates = legalMovesForSelected.filter(
        m => m.to[0] === r && m.to[1] === c
      );

      if (moveCandidates.length > 0) {
        if (moveCandidates.some(m => m.promotion)) {
          setPromotionPending({ from: [sr, sc], to: [r, c] });
          return;
        }
        executePlayerMove(moveCandidates[0], gameState);
        return;
      }

      if (pc === PLAYER_COLOR) {
        const moves = getLegalMoves(gameState, r, c);
        if (moves.length === 0) {
          triggerShake(sq);
          setSelected(null);
          setLegalMovesForSelected([]);
          return;
        }
        setSelected(sq);
        setLegalMovesForSelected(moves);
        return;
      }

      triggerShake(sq);
      setSelected(null);
      setLegalMovesForSelected([]);
      return;
    }

    if (pc === PLAYER_COLOR) {
      const moves = getLegalMoves(gameState, r, c);
      if (moves.length === 0) {
        triggerShake(sq);
        return;
      }
      setSelected(sq);
      setLegalMovesForSelected(moves);
    }
  }, [gameState, selected, legalMovesForSelected, phase, aiThinking, promotionPending, executePlayerMove]);

  const completePromotion = useCallback((pt: 'Q' | 'R' | 'B' | 'N') => {
    if (!promotionPending) return;
    const { from, to } = promotionPending;
    const moves = getLegalMoves(gameState, from[0], from[1]);
    const move = moves.find(m => m.to[0] === to[0] && m.to[1] === to[1] && m.promotion === pt);
    if (move) {
      setPromotionPending(null);
      setSelected(null);
      setLegalMovesForSelected([]);
      executePlayerMove(move, gameState);
    }
  }, [promotionPending, gameState, executePlayerMove]);

  const resetGame = useCallback(() => {
    const initial = parseFEN(INITIAL_FEN);
    stateHistoryRef.current = [initial];
    setGameState(initial);
    setSelected(null);
    setLegalMovesForSelected([]);
    setPhase('playing');
    setHistory([]);
    setCapturedByWhite([]);
    setCapturedByBlack([]);
    setLastMove(null);
    setAiThinking(false);
    setAiDepth(0);
    setAiNodes(0);
    setAiScore(0);
    setPromotionPending(null);
    setShakeSq(null);
    
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  const undoMove = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const hist = stateHistoryRef.current;
    if (hist.length < 3) return;
    stateHistoryRef.current = hist.slice(0, hist.length - 2);
    const prev = stateHistoryRef.current[stateHistoryRef.current.length - 1];
    setGameState(prev);
    setPhase('playing');
    setSelected(null);
    setLegalMovesForSelected([]);
    setHistory(h => h.slice(0, h.length - 2));
    setAiThinking(false);
    setAiDepth(0);
    setAiNodes(0);
    setAiScore(0);
    
    const newHistEntries = history.slice(0, history.length - 2);
    const wb: Piece[] = [], bb: Piece[] = [];
    for (const e of newHistEntries) {
      if (e.capturedPiece) {
        if (e.capturedPiece[0] === 'b') wb.push(e.capturedPiece);
        else bb.push(e.capturedPiece);
      }
    }
    setCapturedByWhite(wb);
    setCapturedByBlack(bb);
    const lastH = newHistEntries[newHistEntries.length - 1];
    setLastMove(lastH?.move ?? null);
  }, [history]);

  const setAIDifficulty = useCallback((timeMs: number) => {
    setAiTimeMs(timeMs);
  }, []);

  const inCheck = isInCheck(gameState, gameState.turn);

  return {
    gameState,
    selected,
    legalMovesForSelected,
    phase,
    history,
    capturedByWhite,
    capturedByBlack,
    lastMove,
    aiThinking,
    aiDepth,
    aiNodes,
    aiScore,
    inCheck,
    promotionPending,
    shakeSq,
    aiInitialized,
    aiInitError,
    selectSquare,
    completePromotion,
    resetGame,
    undoMove,
    setAIDifficulty,
  };
}
